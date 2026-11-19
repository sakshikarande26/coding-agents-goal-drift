#!/usr/bin/env python3
"""Validate finalized chosen/rejected preference pairs (JSON or JSONL).

Use AFTER independent A/B human annotations have been consolidated into a
final preference. Ties and unresolved/uncertain comparisons stay in annotation
files; this validator deliberately does not assign a preference or convert
annotations into training examples.

From the repository root:
    python -m data_eval.validate_pairs examples/pairs/valid_pair.json
    python -m data_eval.validate_pairs pairs/final/ --trajectories rollouts/

Python 3.9+; no external packages. See PAIR_SCHEMA_UPDATE.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

try:
    from .branching import shared_prefix
    from .validate_trajectories import SPLITS, collect_inputs, iter_records
except ImportError:
    from branching import shared_prefix
    from validate_trajectories import SPLITS, collect_inputs, iter_records

DRIFT_TYPES = frozenset({
    "loop", "edits_tests", "unrelated_edit", "scope_creep",
    "hardcode", "skipped_requirement",
})
LABEL_SOURCES = frozenset({"auto", "human"})

PAIR_SCHEMA = {
    "name": "finalized_preference_pair",
    "version": "0.3-multilabel",
    "required_fields": {
        "pair_id": "non-empty string, unique in batch",
        "task_id": "non-empty string",
        "parent_traj_id": "non-empty string linking to parent trajectory",
        "prefix_len": "integer >= 0; completed assistant+tool turns in parent",
        "prompt_messages": "verbatim array of raw shared history message objects",
        "chosen": "non-empty assistant completion (string or raw assistant message object)",
        "rejected": "non-empty assistant completion (string or raw assistant message object)",
        "chosen_drift": "list of distinct drift-type strings; [] means no identified drift",
        "rejected_drift": "list of distinct drift-type strings; [] means no identified drift",
        "label_source": "auto or human",
        "split": "train, heldout, or verified",
    },
    "optional_fields": {
        "chosen_idx": "integer >= 0, must accompany rejected_idx",
        "rejected_idx": "integer >= 0, must accompany chosen_idx",
        "prefix_message_count": "integer >= 0; disambiguates no-tool-call prefix boundaries",
        "chosen_traj_id": "non-empty string for provenance",
        "rejected_traj_id": "non-empty string for provenance",
    },
    "notes": [
        "Preference is encoded by chosen/rejected ordering, not a separate preference field.",
        "Each drift list can contain multiple labels and may be empty.",
        "Do not use the strings 'none' or 'uncertain' in finalized drift lists.",
        "Ties, uncertain judgments, and individual A/B votes belong in annotation data.",
        "The deprecated pair-level drift_type field is rejected to prevent silent data loss.",
    ],
}


def _valid_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_drift_list(value: Any, field: str) -> list[str]:
    """Check FINAL labels; uncertainty must be adjudicated before export."""
    if not isinstance(value, list):
        return [f"{field}: expected a list (use [] for no identified drift)"]
    errors: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{field}[{i}]: expected drift-type string")
        elif item not in DRIFT_TYPES:
            errors.append(f"{field}[{i}]: unknown drift type {item!r}; allowed: {sorted(DRIFT_TYPES)}")
        elif item in seen:
            errors.append(f"{field}[{i}]: duplicate drift type {item!r}")
        else:
            seen.add(item)
    return errors


def validate_pair(pair: Any) -> list[str]:
    """Return all schema errors; never modifies the supplied pair."""
    errors: list[str] = []
    if not isinstance(pair, dict):
        return ["root: expected object"]

    for field in ("pair_id", "task_id", "parent_traj_id"):
        if not _valid_nonempty_text(pair.get(field)):
            errors.append(f"{field}: required non-empty string")

    if not _is_nonnegative_int(pair.get("prefix_len")):
        errors.append("prefix_len: expected integer >= 0")

    prompt = pair.get("prompt_messages")
    if not isinstance(prompt, list):
        errors.append("prompt_messages: expected array")
    else:
        for i, msg in enumerate(prompt):
            if not isinstance(msg, dict) or not _valid_nonempty_text(msg.get("role")):
                errors.append(f"prompt_messages[{i}]: expected message object with role")

    for field in ("chosen", "rejected"):
        val = pair.get(field)
        if isinstance(val, str):
            if not val.strip():
                errors.append(f"{field}: completion cannot be empty")
        elif isinstance(val, dict):
            if val.get("role") != "assistant":
                errors.append(f"{field}: message object must have role='assistant'")
            if not val.get("content") and not val.get("tool_calls") and not val.get("reasoning_content"):
                errors.append(f"{field}: no content, tool calls, or reasoning")
        else:
            errors.append(f"{field}: expected non-empty string or assistant message object")

    if "chosen" in pair and "rejected" in pair and pair["chosen"] == pair["rejected"]:
        errors.append("chosen and rejected: cannot be identical")

    for field in ("chosen_drift", "rejected_drift"):
        errors.extend(_validate_drift_list(pair.get(field), field))

    if "drift_type" in pair:
        errors.append("drift_type: deprecated single pair label; use chosen_drift and rejected_drift lists")
    if "preference" in pair:
        errors.append("preference: A/B/tie/uncertain belongs in human annotations, not finalized chosen/rejected pairs")
    if not isinstance(pair.get("label_source"), str) or pair["label_source"] not in LABEL_SOURCES:
        errors.append(f"label_source: must be one of {sorted(LABEL_SOURCES)}")
    if not isinstance(pair.get("split"), str) or pair["split"] not in SPLITS:
        errors.append(f"split: must be one of {sorted(SPLITS)}")

    if "prefix_message_count" in pair and not _is_nonnegative_int(pair["prefix_message_count"]):
        errors.append("prefix_message_count: expected integer >= 0")
    for field in ("chosen_traj_id", "rejected_traj_id"):
        if field in pair and not _valid_nonempty_text(pair[field]):
            errors.append(f"{field}: expected nonempty string")
    for field in ("chosen_idx", "rejected_idx"):
        if field in pair and not _is_nonnegative_int(pair[field]):
            errors.append(f"{field}: expected integer >= 0")
    if ("chosen_idx" in pair) != ("rejected_idx" in pair):
        errors.append("chosen_idx and rejected_idx: supply both or neither")
    if ("chosen_idx" in pair and "rejected_idx" in pair and
            pair["chosen_idx"] == pair["rejected_idx"]):
        errors.append("chosen_idx and rejected_idx: must differ")
    return errors


def _first_command(value: Any) -> str | None:
    """Compare single-command choices, ignoring incidental tool-call IDs."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        calls = value.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        call = calls[0]
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return None
        args = call["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if isinstance(args, dict) and isinstance(args.get("command"), str):
            return args["command"].strip()
    return None


def pair_warnings(pair: Any) -> list[str]:
    if not isinstance(pair, dict):
        return []
    a, b = _first_command(pair.get("chosen")), _first_command(pair.get("rejected"))
    if a and b and a == b:
        return ["chosen and rejected execute the same command; only text/tool metadata differs"]
    return []


def load_parent_map(source: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in collect_inputs(source):
        if path.name in {"pair_manifest.json", "drift_flags.json"}:
            continue
        for location, record, parse_err in iter_records(path):
            if parse_err:
                raise ValueError(f"{location}: {parse_err}")
            if isinstance(record, dict) and isinstance(record.get("traj_id"), str):
                if record["traj_id"] in by_id:
                    raise ValueError(f"Duplicate parent trajectory ID: {record['traj_id']}")
                by_id[record["traj_id"]] = record
    return by_id


def validate_pairs(source: Path, *, trajectories: Path | None = None) -> dict:
    """Validate a batch and, optionally, the exact parent prefix for each pair."""
    parents = load_parent_map(trajectories) if trajectories is not None else None
    files = [p for p in collect_inputs(source) if p.name not in {"pair_manifest.json", "manifest.json"}]
    seen_ids: dict[str, str] = {}
    counts = Counter()
    chosen_types = Counter()
    rejected_types = Counter()
    splits = Counter()
    issues: list[dict] = []

    for path in files:
        try:
            for location, rec, parse_error in iter_records(path):
                counts["records_scanned"] += 1
                errors = [parse_error] if parse_error else validate_pair(rec)
                warnings = [] if parse_error else pair_warnings(rec)
                if isinstance(rec, dict):
                    ident = rec.get("pair_id")
                    if _valid_nonempty_text(ident):
                        if ident in seen_ids:
                            errors.append(f"pair_id: duplicate of {seen_ids[ident]}")
                        else:
                            seen_ids[ident] = location
                    if parents is not None and _valid_nonempty_text(rec.get("parent_traj_id")):
                        parent = parents.get(rec["parent_traj_id"])
                        if parent is None:
                            errors.append("parent_traj_id: not found in provided trajectories")
                        else:
                            if rec.get("task_id") != parent.get("task_id"):
                                errors.append("task_id: does not match parent trajectory")
                            if rec.get("split") != parent.get("split"):
                                errors.append("split: does not match parent trajectory")
                            if _is_nonnegative_int(rec.get("prefix_len")):
                                expected, err = shared_prefix(
                                    parent.get("messages"), rec["prefix_len"],
                                    prefix_message_count=rec.get("prefix_message_count"),
                                )
                                if err:
                                    errors.append(f"parent prompt prefix: {err}")
                                elif rec.get("prompt_messages") != expected:
                                    errors.append("prompt_messages: not the exact parent history at prefix_len")
                if errors:
                    counts["invalid_pairs"] += 1
                else:
                    counts["valid_pairs"] += 1
                    chosen_types.update(rec["chosen_drift"])
                    rejected_types.update(rec["rejected_drift"])
                    splits[rec["split"]] += 1
                counts["warnings"] += len(warnings)
                if errors or warnings:
                    issues.append({
                        "location": location,
                        "pair_id": rec.get("pair_id") if isinstance(rec, dict) else None,
                        "errors": errors,
                        "warnings": warnings,
                    })
        except (UnicodeError, OSError) as exc:
            counts["records_scanned"] += 1
            counts["invalid_pairs"] += 1
            issues.append({"location": str(path), "pair_id": None,
                           "errors": [f"File read error: {exc}"], "warnings": []})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "trajectory_source": str(trajectories) if trajectories else None,
        "schema": PAIR_SCHEMA,
        "counts": {
            "files_scanned": len(files), "records_scanned": counts["records_scanned"],
            "valid_pairs": counts["valid_pairs"], "invalid_pairs": counts["invalid_pairs"],
            "warnings": counts["warnings"],
        },
        "valid_by_chosen_drift_type": dict(sorted(chosen_types.items())),
        "valid_by_rejected_drift_type": dict(sorted(rejected_types.items())),
        "valid_by_split": dict(sorted(splits.items())),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Finalized pair JSON/JSONL file or directory")
    parser.add_argument("--trajectories", type=Path,
                        help="Optional parent trajectory JSON/JSONL files for exact prefix comparison")
    parser.add_argument("--manifest", type=Path, help="Output path for pair_manifest.json")
    args = parser.parse_args()
    try:
        report = validate_pairs(args.input, trajectories=args.trajectories)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    out = args.manifest or ((args.input if args.input.is_dir() else args.input.parent) / "pair_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Pairs:", report["counts"])
    for issue in report["issues"]:
        print(issue["location"])
        for error in issue["errors"]:
            print("  ERROR:", error)
        for warning in issue.get("warnings", []):
            print("  WARNING:", warning)
    print("Manifest saved to:", out)
    return 1 if report["counts"]["invalid_pairs"] else 0


if __name__ == "__main__":
    sys.exit(main())
