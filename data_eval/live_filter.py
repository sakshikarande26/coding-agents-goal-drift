#!/usr/bin/env python3
"""Live rule-filter baseline: score proposed commands *before* execution.

Import in the agent's selector:
    from evaluation.live_filter import select_candidate
    result = select_candidate(candidates, executed_steps, problem_statement)
    next_command = result["chosen_command"]

For local testing (no agent or sandbox required):
    python evaluation/live_filter.py --demo
    python evaluation/live_filter.py --trajectory examples/stand_in_trajectory.json \
        --candidate "find src -name '*.py'" --candidate "ls -la"

``history`` may be a derived list from drift_rules.extract_steps, or raw
mini-SWE-agent messages[]. The tool must be hooked up by Track 1; importing
this module does not automatically change the agent's decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .drift_rules import evaluate_candidate, extract_steps
else:
    from drift_rules import evaluate_candidate, extract_steps


def select_candidate(
    candidates: list[str],
    history: list[dict[str, Any]],
    problem_statement: str,
) -> dict[str, Any]:
    """Select a candidate by fixed penalty, using only pre-action information.

    A high-severity flag marks a candidate ``blocked`` (advisory). If one or
    more candidates are not blocked, choose the lowest-penalty nonblocked one.
    If ALL candidates are blocked, choose the least-penalized candidate anyway
    and mark ``fallback_used``. The caller may opt to regenerate candidates.
    Stable tie break: candidate position in the input list.
    """
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list of command strings")
    if any(not isinstance(c, str) or not c.strip() for c in candidates):
        raise ValueError("every candidate must be a nonempty command string")

    evaluations = [evaluate_candidate(c, history, problem_statement) for c in candidates]
    eligible = [i for i, item in enumerate(evaluations) if not item["blocked"]]
    fallback_used = not bool(eligible)
    pool = eligible if eligible else list(range(len(evaluations)))
    chosen_idx = min(pool, key=lambda i: (evaluations[i]["penalty"], i))
    return {
        "chosen_idx": chosen_idx,
        "chosen_command": candidates[chosen_idx],
        "fallback_used": fallback_used,
        "all_candidates_blocked": fallback_used,
        "evaluations": evaluations,
    }


def _load_single_trajectory(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise ValueError("Expected a trajectory JSON object with messages[]")
    return record


def _synthetic_demo() -> tuple[list[str], list[dict[str, Any]], str]:
    issue = "Fix URL routing in router.py."
    # Previous two commands failed identically; an identical next one is loop drift.
    history = [
        {"command": "find src -name '*.py'", "observation": "find: 'src': No such file or directory"},
        {"command": "find src -name '*.py'", "observation": "find: 'src': No such file or directory"},
    ]
    candidates = [
        "find src -name '*.py'",                  # loop -> high
        "sed -i 's/x/y/' tests/test_router.py",    # edits_tests -> high
        "sed -i 's/x/y/' billing.py",              # unrelated_edit -> warning
        "ls -la",                                  # no flags -> should win
    ]
    return candidates, history, issue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run embedded synthetic candidate example")
    parser.add_argument("--trajectory", type=Path, help="JSON trajectory with messages[] history")
    parser.add_argument("--candidate", action="append", help="Possible next command (repeat for several)")
    parser.add_argument("--output", type=Path, help="Write JSON result to file instead of stdout")
    args = parser.parse_args()
    if args.demo:
        if args.trajectory or args.candidate:
            parser.error("--demo cannot be combined with --trajectory or --candidate")
        candidates, history, issue = _synthetic_demo()
    else:
        if not args.trajectory or not args.candidate:
            parser.error("provide --demo, or --trajectory plus one or more --candidate")
        record = _load_single_trajectory(args.trajectory)
        history = extract_steps(record["messages"])
        issue = record.get("problem_statement", "")
        candidates = args.candidate

    result = select_candidate(candidates, history, issue)
    content = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Live filter result written to {args.output}")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
