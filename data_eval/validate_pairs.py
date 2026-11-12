#!/usr/bin/env python3
"""Validate preference-pair JSON/JSONL files, optionally matching trajectory parents.

From project root:
  python -m data_eval.validate_pairs examples/pairs/valid_pair.json
  python -m data_eval.validate_pairs pairs/v1/ --trajectories rollouts/v1/

No external Python packages. The pair contract is v0/provisional: coordinate
exact chosen/rejected representation and drift_type labels with Track 3.
"""
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

DRIFT_TYPES = {"loop", "edits_tests", "unrelated_edit", "scope_creep", "hardcode", "skipped_requirement"}
LABEL_SOURCES = {"auto", "human"}

PAIR_SCHEMA = {
    "name": "preference_pair",
    "version": "0.1-proposed",
    "required_fields": {
        "pair_id": "non-empty string; unique in batch",
        "task_id": "non-empty string",
        "parent_traj_id": "non-empty string; links to parent trajectory",
        "prefix_len": "integer >= 0; completed assistant+tool pairs",
        "prompt_messages": "array of raw prompt/history message objects",
        "chosen": "non-empty assistant completion (string or raw message object)",
        "rejected": "non-empty assistant completion (string or raw message object)",
        "label_source": "auto or human",
        "drift_type": "one of six project drift types or 'none'",
        "split": "train, heldout, verified",
    },
    "optional_fields": {"chosen_idx": "integer >= 0", "rejected_idx": "integer >= 0"}
}


def validate_pair(pair: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(pair, dict):
        return ["root: expected object"]
    for f in ("pair_id", "task_id", "parent_traj_id"):
        val = pair.get(f)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{f}: required non-empty string")
    n = pair.get("prefix_len")
    if type(n) is not int or n < 0:
        errors.append("prefix_len: expected integer >= 0")
    prompt = pair.get("prompt_messages")
    if not isinstance(prompt, list):
        errors.append("prompt_messages: expected array")
    else:
        for i, msg in enumerate(prompt):
            if not isinstance(msg, dict) or not isinstance(msg.get("role"), str) or not msg["role"]:
                errors.append(f"prompt_messages[{i}]: expected message object with role")
    for f in ("chosen", "rejected"):
        val = pair.get(f)
        if isinstance(val, str):
            if not val.strip():
                errors.append(f"{f}: completion cannot be empty")
        elif isinstance(val, dict):
            if val.get("role") != "assistant":
                errors.append(f"{f}: message object must have role='assistant'")
            if not val.get("content") and not val.get("tool_calls") and not val.get("reasoning_content"):
                errors.append(f"{f}: no content, tool calls, or reasoning")
        else:
            errors.append(f"{f}: expected non-empty string or assistant message object")
    if "chosen" in pair and "rejected" in pair and pair["chosen"] == pair["rejected"]:
        errors.append("chosen and rejected: cannot be identical")
    if pair.get("label_source") not in LABEL_SOURCES:
        errors.append(f"label_source: must be one of {sorted(LABEL_SOURCES)}")
    if pair.get("drift_type") not in DRIFT_TYPES | {"none"}:
        errors.append(f"drift_type: must be one of {sorted(DRIFT_TYPES | {'none'})}")
    if pair.get("split") not in SPLITS:
        errors.append(f"split: must be one of {sorted(SPLITS)}")
    for f in ("chosen_idx", "rejected_idx"):
        if f in pair and (type(pair[f]) is not int or pair[f] < 0):
            errors.append(f"{f}: expected integer >= 0")
    if ("chosen_idx" in pair and "rejected_idx" in pair and
            pair["chosen_idx"] == pair["rejected_idx"]):
        errors.append("chosen_idx and rejected_idx: must differ")
    return errors


def load_parent_map(source: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in collect_inputs(source):
        for location, record, parse_err in iter_records(path):
            if parse_err:
                raise ValueError(f"{location}: {parse_err}")
            if isinstance(record, dict) and isinstance(record.get("traj_id"), str):
                if record["traj_id"] in by_id:
                    raise ValueError(f"Duplicate parent trajectory ID: {record['traj_id']}")
                by_id[record["traj_id"]] = record
    return by_id


def validate_pairs(source: Path, *, trajectories: Path | None = None) -> dict:
    parents = load_parent_map(trajectories) if trajectories is not None else None
    files = collect_inputs(source)
    # Manifest files are outputs, not pair records.
    files = [p for p in files if p.name not in {"pair_manifest.json", "manifest.json"}]
    seen_ids: dict[str, str] = {}
    counts = Counter()
    types = Counter()
    splits = Counter()
    issues: list[dict] = []
    for path in files:
        try:
            for location, rec, parse_error in iter_records(path):
                counts["records_scanned"] += 1
                errors = [parse_error] if parse_error else validate_pair(rec)
                if isinstance(rec, dict):
                    ident = rec.get("pair_id")
                    if isinstance(ident, str) and ident.strip():
                        if ident in seen_ids:
                            errors.append(f"pair_id: duplicate of {seen_ids[ident]}")
                        else:
                            seen_ids[ident] = location
                    if parents is not None and not any(x.startswith("parent_traj_id:") for x in errors):
                        parent = parents.get(rec.get("parent_traj_id"))
                        if parent is None:
                            errors.append("parent_traj_id: not found in provided trajectories")
                        else:
                            if rec.get("task_id") != parent.get("task_id"):
                                errors.append("task_id: does not match parent trajectory")
                            if rec.get("split") != parent.get("split"):
                                errors.append("split: does not match parent trajectory")
                            if type(rec.get("prefix_len")) is int and rec["prefix_len"] >= 0:
                                expected, err = shared_prefix(parent.get("messages"), rec["prefix_len"])
                                if err:
                                    errors.append(f"parent prompt prefix: {err}")
                                elif rec.get("prompt_messages") != expected:
                                    errors.append("prompt_messages: not the exact parent history at prefix_len")
                if errors:
                    counts["invalid_pairs"] += 1
                    issues.append({"location": location, "pair_id": rec.get("pair_id") if isinstance(rec, dict) else None,
                                   "errors": errors})
                else:
                    counts["valid_pairs"] += 1
                    types[rec["drift_type"]] += 1
                    splits[rec["split"]] += 1
        except (UnicodeError, OSError) as exc:
            counts["records_scanned"] += 1
            counts["invalid_pairs"] += 1
            issues.append({"location": str(path), "pair_id": None, "errors": [f"File read error: {exc}"]})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "trajectory_source": str(trajectories) if trajectories else None,
        "schema": PAIR_SCHEMA,
        "counts": {"files_scanned": len(files), "records_scanned": counts["records_scanned"],
                   "valid_pairs": counts["valid_pairs"], "invalid_pairs": counts["invalid_pairs"]},
        "valid_by_drift_type": dict(sorted(types.items())),
        "valid_by_split": dict(sorted(splits.items())),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Pair JSON/JSONL file or directory")
    parser.add_argument("--trajectories", type=Path,
                        help="Optional trajectory JSON/JSONL files for exact parent prompt checks")
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
    print("Manifest saved to:", out)
    return 1 if report["counts"]["invalid_pairs"] else 0


if __name__ == "__main__":
    sys.exit(main())
