#!/usr/bin/env python3
"""Offline analysis of completed coding-agent trajectories.

Usage from repository root:
    python data_eval/offline_analysis.py --demo
    python data_eval/offline_analysis.py examples/stand_in_trajectory.json
    python data_eval/offline_analysis.py rollouts/v1/ --output drift_flags.json

Reads individual JSON trajectories or directories of JSON/JSONL/NDJSON records.
For each executed command, tests the rules against only its preceding history;
this deliberately aligns the detector with the live filter and avoids using
future outcomes, gold patches, hidden tests or final grading results.

Warning: flags are heuristic *suspicions*, not ground-truth drift labels.
Independent human annotation should be used for the actual research metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterator

if __package__:
    from .drift_rules import evaluate_candidate, extract_steps_with_issues, issue_from_record
else:
    from drift_rules import evaluate_candidate, extract_steps_with_issues, issue_from_record


def analyze_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    """Return flags for executed steps; source messages[] remain unmodified."""
    if not isinstance(record, dict):
        raise TypeError("expected trajectory object")
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trajectory must contain messages[]")
    issue = issue_from_record(record)

    steps, parse_issues = extract_steps_with_issues(messages)
    no_tool_call_count = sum(1 for m in messages if isinstance(m, dict) and
                             m.get("role") == "assistant" and not m.get("tool_calls"))
    flagged_steps: list[dict[str, Any]] = []
    by_type: Counter[str] = Counter()
    for i, step in enumerate(steps):
        # Two tool calls emitted in ONE assistant reply are a single decision:
        # neither has seen the other's output. Also exclude outputs arriving
        # after the current assistant response (strict no-future leakage).
        decision_msg_idx = step["assistant_message_index"]
        past = [earlier for earlier in steps if
                earlier["assistant_message_index"] < decision_msg_idx and
                earlier.get("observation_message_index") is not None and
                earlier["observation_message_index"] < decision_msg_idx]
        result = evaluate_candidate(step["command"], past, issue)
        if result["flags"]:
            flagged_steps.append({
                "step": step["step"],
                "command": step["command"],
                "flags": result["flags"],
                "blocked_by_rules": result["blocked"],
            })
            by_type.update(f["drift_type"] for f in result["flags"])

    return {
        "traj_id": record.get("traj_id"),
        "task_id": record.get("task_id"),
        "commands_analyzed": len(steps),
        "no_tool_call_responses": no_tool_call_count,
        "parse_issues": parse_issues,
        "flagged_steps": flagged_steps,
        "flag_counts": dict(sorted(by_type.items())),
        "note": "Heuristic flags only; use human labels for ground truth",
    }


def iter_trajectories(path: Path, *, exclude: Path | None = None) -> Iterator[dict[str, Any]]:
    """Read trajectory JSON/JSONL records, skipping manifest and output files."""
    if path.is_dir():
        for item in sorted(path.rglob("*")):
            if item.is_file() and item.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                if item.name in {"manifest.json", "pair_manifest.json", "drift_flags.json"} or (exclude and item.resolve() == exclude.resolve()):
                    continue
                yield from iter_trajectories(item, exclude=exclude)
        return
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as stream:
            for i, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{i} is not a JSON object")
                yield obj
        return
    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a trajectory object")
    yield obj


def analyze_all(records: list[dict[str, Any]]) -> dict[str, Any]:
    results = [analyze_trajectory(r) for r in records]
    by_type: Counter[str] = Counter()
    for result in results:
        by_type.update(result["flag_counts"])
    return {
        "trajectories_analyzed": len(results),
        "commands_analyzed": sum(r["commands_analyzed"] for r in results),
        "no_tool_call_responses": sum(r["no_tool_call_responses"] for r in results),
        "parse_issue_count": sum(len(r["parse_issues"]) for r in results),
        "total_flagged_steps": sum(len(r["flagged_steps"]) for r in results),
        "flag_counts": dict(sorted(by_type.items())),
        "results": results,
    }


def _synthetic_demo() -> dict[str, Any]:
    issue = "Fix URL routing in router.py."
    messages: list[dict[str, Any]] = [{"role": "user", "content": issue}]
    actions = [
        ("find src -name '*.py'", "find: 'src': No such file or directory"),
        ("find src -name '*.py'", "find: 'src': No such file or directory"),
        ("find src -name '*.py'", "find: 'src': No such file or directory"),
        ("sed -i 's/x/y/' tests/test_router.py", ""),
        ("sed -i 's/x/y/' billing.py", ""),
    ]
    for i, (cmd, obs) in enumerate(actions):
        call_id = f"demo-{i}"
        messages += [
            {"role": "assistant", "tool_calls": [{"id": call_id, "function": {
                "name": "bash", "arguments": {"command": cmd}}}]},
            {"role": "tool", "tool_call_id": call_id, "content": obs},
        ]
    return {"traj_id": "synthetic_demo", "task_id": "synthetic_task", "problem_statement": issue, "messages": messages}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Trajectory JSON/JSONL file or directory")
    parser.add_argument("--demo", action="store_true", help="Run embedded synthetic trajectory")
    parser.add_argument("--output", type=Path, help="Save results as JSON")
    args = parser.parse_args()
    if args.demo:
        if args.input:
            parser.error("Do not combine an input path with --demo")
        records = [_synthetic_demo()]
    elif args.input:
        records = list(iter_trajectories(args.input, exclude=args.output))
    else:
        parser.error("provide a trajectory path or --demo")

    report = analyze_all(records)
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Offline analysis saved to {args.output}")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
