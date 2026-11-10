#!/usr/bin/env python3
"""Validate trajectory records and create a manifest of rollout counts.

Supports one trajectory per .json file or one trajectory per line in .jsonl/.ndjson.
The required fields match the agreed schema.

Usage:
    python validate_trajectories.py rollouts/v1
    python validate_trajectories.py rollouts/v1 --manifest rollouts/v1/manifest.json
    python validate_trajectories.py example_trajectory.json

Requires Python 3.9+; no third-party packages.
Exit status is 0 if all records are valid and 1 if any are invalid.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


SPLITS = {"train", "heldout", "verified"}
MATCHES_PATTERN = re.compile(r"^(\d+)/(\d+)$")

TRAJECTORY_SCHEMA = {
    "name": "trajectory",
    "version": "1.1",
    "required_fields": {
        "traj_id": "non-empty string; unique",
        "task_id": "non-empty string",
        "split": "train, heldout, or verified",
        "repo": "non-empty string",
        "docker_image": "non-empty string",
        "problem_statement": "non-empty string",
        "condition": "non-empty string",
        "model": "non-empty string",
        "sampling": {
            "temperature": "number >= 0",
            "top_p": "number > 0 and <= 1",
            "presence_penalty": "number",
            "seed": "integer"
        },
        "thinking": "boolean",
        "step_limit": "integer >= 1",
        "versions": {
            "model_revision": "non-empty string (exact model revision)",
            "chat_template_hash": "non-empty string (hash of chat template)",
            "other_fields": "optional version strings"
        },
        "finished_at": "ISO 8601 timestamp",
        "exit_status": "string or null",
        "steps": "integer >= 0",
        "resolved": "boolean",
        "tests_matched": "matched/total string",
        "test_summary": "string or object",
        "patch": "string",
        "files_touched": "array of strings",
        "ref_patch_files": "array of strings",
        "parent_traj_id": "non-empty string or null; null if not branched",
        "branch_step": "integer >= 0 or null; null if not branched",
        "messages": "verbatim raw mini-swe-agent history (array of message objects; do not truncate)"
    }
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_trajectory(record: Any) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one parsed trajectory object.

    Errors break the agreed data contract. Warnings highlight possible
    inconsistencies without assuming undocumented mini-SWE-agent internals.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(record, dict):
        return ["root: expected a JSON object"], warnings

    def require(field: str, kind: type, *, nullable: bool = False, allow_blank: bool = False) -> Any:
        if field not in record:
            errors.append(f"{field}: missing required field")
            return None
        val = record[field]
        if nullable and val is None:
            return None
        ok = (_is_int(val) if kind is int else
              _is_number(val) if kind is float else
              type(val) is bool if kind is bool else
              isinstance(val, kind))
        if not ok:
            expected = "number" if kind is float else kind.__name__
            errors.append(f"{field}: expected {expected}, got {type(val).__name__}")
            return None
        if kind is str and not allow_blank and not val.strip():
            errors.append(f"{field}: must not be blank")
        return val

    for field in (
        "traj_id", "task_id", "repo", "docker_image", "condition",
        "model", "problem_statement", "finished_at",
    ):
        require(field, str)

    split = require("split", str)
    if isinstance(split, str) and split not in SPLITS:
        errors.append(f"split: must be one of {sorted(SPLITS)}")

    require("thinking", bool)
    require("resolved", bool)
    require("exit_status", str, nullable=True)  # null = no recorded exit message
    require("patch", str, allow_blank=True)    # empty patch is valid

    step_limit = require("step_limit", int)
    steps = require("steps", int)  # count, NOT an array in this version
    if _is_int(step_limit) and step_limit < 1:
        errors.append("step_limit: must be >= 1")
    if _is_int(steps) and steps < 0:
        errors.append("steps: must be >= 0")
    if _is_int(steps) and _is_int(step_limit) and steps > step_limit:
        warnings.append("steps exceeds step_limit; check how steps are counted")

    # A branch must identify both its parent and the point where it branched.
    # Non-branched trajectories explicitly use null for both fields.
    parent_traj_id = require("parent_traj_id", str, nullable=True)
    branch_step = require("branch_step", int, nullable=True)
    if (parent_traj_id is None) != (branch_step is None):
        # If either is missing, require() already records the missing-field error.
        if "parent_traj_id" in record and "branch_step" in record:
            errors.append("parent_traj_id and branch_step: must both be null or both be set")
    if _is_int(branch_step) and branch_step < 0:
        errors.append("branch_step: must be >= 0")
    if isinstance(parent_traj_id, str) and parent_traj_id == record.get("traj_id"):
        errors.append("parent_traj_id: cannot equal traj_id")

    versions = require("versions", dict)
    if isinstance(versions, dict):
        for name, version in versions.items():
            if not isinstance(name, str) or not isinstance(version, str):
                errors.append("versions: expected mapping of name -> version/hash string")
                break
        for field in ("model_revision", "chat_template_hash"):
            if field not in versions:
                errors.append(f"versions.{field}: missing required field")
            elif not isinstance(versions[field], str) or not versions[field].strip():
                errors.append(f"versions.{field}: expected non-empty string")

    ts = record.get("finished_at")
    if isinstance(ts, str) and ts.strip():
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            errors.append("finished_at: expected ISO 8601 timestamp")

    sampling = require("sampling", dict)
    if isinstance(sampling, dict):
        for name in ("temperature", "top_p", "presence_penalty"):
            if name not in sampling:
                errors.append(f"sampling.{name}: missing required field")
            elif not _is_number(sampling[name]):
                errors.append(f"sampling.{name}: expected number")
        if "seed" not in sampling:
            errors.append("sampling.seed: missing required field")
        elif not _is_int(sampling["seed"]):
            errors.append("sampling.seed: expected integer")
        temp, top_p = sampling.get("temperature"), sampling.get("top_p")
        if _is_number(temp) and temp < 0:
            errors.append("sampling.temperature: must be >= 0")
        if _is_number(top_p) and not (0 < top_p <= 1):
            errors.append("sampling.top_p: must be in (0, 1]")

    tests_matched = require("tests_matched", str)
    if isinstance(tests_matched, str):
        match = MATCHES_PATTERN.fullmatch(tests_matched)
        if not match:
            errors.append("tests_matched: expected 'matched/total', e.g. '13/14'")
        elif int(match[1]) > int(match[2]):
            errors.append("tests_matched: matched cannot exceed total")

    # Sakshi's existing notebook returns a string summary; supporting an object
    # also allows future structured grading summaries without schema breakage.
    if "test_summary" not in record:
        errors.append("test_summary: missing required field")
    elif not isinstance(record["test_summary"], (str, dict)):
        errors.append("test_summary: expected string or object")

    for field in ("files_touched", "ref_patch_files"):
        paths = require(field, list)
        if isinstance(paths, list):
            for i, path in enumerate(paths):
                if not isinstance(path, str) or not path.strip():
                    errors.append(f"{field}[{i}]: expected non-empty file path string")
            if len(paths) != len(set(p for p in paths if isinstance(p, str))):
                warnings.append(f"{field}: contains duplicate paths")

    messages = require("messages", list)
    if isinstance(messages, list):
        assistant_count = 0
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"messages[{i}]: expected object")
                continue
            role = message.get("role")
            if not isinstance(role, str) or not role:
                errors.append(f"messages[{i}].role: expected non-empty string")
                continue
            if role == "assistant":
                assistant_count += 1
                # Not every assistant message necessarily calls a tool.
                calls = message.get("tool_calls")
                if calls is not None:
                    if not isinstance(calls, list):
                        errors.append(f"messages[{i}].tool_calls: expected list")
                    else:
                        for j, call in enumerate(calls):
                            if not isinstance(call, dict):
                                errors.append(f"messages[{i}].tool_calls[{j}]: expected object")
                                continue
                            func = call.get("function")
                            if not isinstance(func, dict):
                                errors.append(f"messages[{i}].tool_calls[{j}].function: expected object")
                                continue
                            args = func.get("arguments")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    errors.append(f"messages[{i}].tool_calls[{j}].function.arguments: invalid JSON string")
                                    continue
                            if not isinstance(args, dict):
                                errors.append(f"messages[{i}].tool_calls[{j}].function.arguments: expected object or JSON-encoded object")
                            elif not isinstance(args.get("command"), str):
                                errors.append(f"messages[{i}].tool_calls[{j}].function.arguments.command: expected string")
            if role == "tool" and "content" not in message:
                errors.append(f"messages[{i}].content: tool output missing")
        if _is_int(steps) and assistant_count != steps:
            warnings.append(f"steps={steps} but found {assistant_count} assistant messages")

    # Additional per-decision data (candidates/scores/chosen_idx) is allowed,
    # but not required until the team finalizes reranking log structure.
    return errors, warnings


def collect_inputs(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path not found: {source}")
    return sorted(p for p in source.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson"}
                  and p.name != "manifest.json")


def iter_records(path: Path):
    """Yield (location, parsed_record, parse_error) for all data in a file."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield f"{path}:{line_no}", json.loads(line), None
                except json.JSONDecodeError as exc:
                    yield f"{path}:{line_no}", None, f"invalid JSON: {exc.msg}"
    else:
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
            yield str(path), obj, None
        except json.JSONDecodeError as exc:
            yield str(path), None, f"invalid JSON: {exc.msg} (line {exc.lineno})"


def create_manifest(source: Path) -> dict:
    files = collect_inputs(source)
    counters = {name: Counter() for name in ("split", "condition", "model", "seed", "exit_status")}
    issues = []
    seen_ids: dict[str, str] = {}
    unique_tasks: set[str] = set()
    n_records = n_valid = n_invalid = n_warnings = 0

    for path in files:
        try:
            for location, record, parse_error in iter_records(path):
                n_records += 1
                errors, warnings = ([parse_error], []) if parse_error else validate_trajectory(record)
                traj_id = record.get("traj_id") if isinstance(record, dict) else None
                task_id = record.get("task_id") if isinstance(record, dict) else None
                if isinstance(traj_id, str) and traj_id.strip():
                    if traj_id in seen_ids:
                        errors.append(f"traj_id: duplicate of record in {seen_ids[traj_id]}")
                    else:
                        seen_ids[traj_id] = location

                if errors:
                    n_invalid += 1
                else:
                    n_valid += 1
                    unique_tasks.add(record["task_id"])
                    for name in ("split", "condition", "model"):
                        counters[name][record[name]] += 1
                    counters["seed"][str(record["sampling"]["seed"])] += 1
                    counters["exit_status"][str(record["exit_status"])] += 1

                n_warnings += len(warnings)
                if errors or warnings:
                    issues.append({"location": location, "traj_id": traj_id,
                                   "task_id": task_id, "errors": errors, "warnings": warnings})
        except (OSError, UnicodeError) as exc:
            n_invalid += 1
            n_records += 1
            issues.append({"location": str(path), "traj_id": None, "task_id": None,
                           "errors": [f"Could not read file: {exc}"], "warnings": []})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "schema": TRAJECTORY_SCHEMA,
        "counts": {
            "files_scanned": len(files),
            "records_scanned": n_records,
            "valid_trajectories": n_valid,
            "invalid_trajectories": n_invalid,
            "unique_valid_tasks": len(unique_tasks),
            "warnings": n_warnings,
        },
        "valid_by_split": dict(sorted(counters["split"].items())),
        "valid_by_condition": dict(sorted(counters["condition"].items())),
        "valid_by_model": dict(sorted(counters["model"].items())),
        "valid_by_seed": dict(sorted(counters["seed"].items())),
        "valid_by_exit_status": dict(sorted(counters["exit_status"].items())),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trajectory .json/.jsonl file or a folder")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Where to write manifest.json (default: inside input folder)")
    args = parser.parse_args()
    try:
        manifest = create_manifest(args.input)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    out = args.manifest or ((args.input if args.input.is_dir() else args.input.parent) / "manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    counts = manifest["counts"]
    print(f"Scanned {counts['files_scanned']} file(s), {counts['records_scanned']} trajectory record(s)")
    print(f"Valid: {counts['valid_trajectories']}  Invalid: {counts['invalid_trajectories']}"
          f"  Warnings: {counts['warnings']}")
    print(f"Unique tasks (valid): {counts['unique_valid_tasks']}")
    for issue in manifest["issues"]:
        print(f"\n{issue['location']}")
        for error in issue["errors"]:
            print(f"  ERROR: {error}")
        for warning in issue["warnings"]:
            print(f"  WARNING: {warning}")
    print(f"\nManifest saved to: {out}")
    return 1 if counts["invalid_trajectories"] else 0


if __name__ == "__main__":
    sys.exit(main())
