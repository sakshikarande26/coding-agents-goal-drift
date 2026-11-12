#!/usr/bin/env python3
"""Live rule-filter baseline: score proposed commands *before* execution.

Import in the agent's selector:
    from data_eval.live_filter import select_candidate
    result = select_candidate(candidates, executed_steps, problem_statement)
    next_command = result["chosen_command"]

For local testing (no agent or sandbox required):
    python data_eval/live_filter.py --demo
    python data_eval/live_filter.py --trajectory examples/stand_in_trajectory.json \
        --candidate "find src -name '*.py'" --candidate "ls -la"

``history`` may be a derived list from drift_rules.extract_steps, or raw
mini-SWE-agent messages[]. The tool must be hooked up by Track 1; importing
this module does not automatically change the agent's decisions.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

if __package__:
    from .drift_rules import evaluate_candidate, extract_steps, issue_from_record
else:
    from drift_rules import evaluate_candidate, extract_steps, issue_from_record


def _commands_from_candidate(candidate: Any) -> list[str]:
    """Read a shell command or full assistant completion, preserving originals."""
    if isinstance(candidate, str):
        return [candidate] if candidate.strip() else []
    if not isinstance(candidate, dict) or candidate.get("role") != "assistant":
        raise TypeError("candidate must be a command string or raw assistant message")
    calls = candidate.get("tool_calls")
    if calls is None or calls == []:
        return []
    if not isinstance(calls, list):
        raise ValueError("assistant candidate tool_calls must be a list")
    commands = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError("candidate tool call must contain a function object")
        if call["function"].get("name", "bash") != "bash":
            raise ValueError("candidate uses unsupported non-bash tool")
        args = call["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ValueError("candidate has invalid tool-call JSON arguments") from exc
        if not isinstance(args, dict) or not isinstance(args.get("command"), str):
            raise ValueError("candidate tool-call arguments must contain command string")
        commands.append(args["command"])
    return commands


def select_candidate(
    candidates: list[str | dict[str, Any]],
    history: list[dict[str, Any]],
    problem_statement: str,
    *,
    tie_break_seed: int = 0,
) -> dict[str, Any]:
    """Score candidates before execution, returning original chosen completion.

    Accepts both command strings (demo) and raw assistant response objects.
    A no-tool-call candidate is non-executable; if *all* candidates are such,
    returns chosen_idx=None and requires_regeneration=True. The caller MUST
    not execute a null chosen_command. Ties are broken with a seeded RNG.

    history must be pre-action; either extracted steps or raw messages[].
    """
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    if not isinstance(history, list):
        raise TypeError("history must be a list")
    if type(tie_break_seed) is not int:
        raise TypeError("tie_break_seed must be integer")
    if history and isinstance(history[0], dict) and "role" in history[0]:
        history = extract_steps(history)
    evaluations = []
    for candidate in candidates:
        try:
            commands = _commands_from_candidate(candidate)
            parse_error = None
        except (TypeError, ValueError) as exc:
            commands, parse_error = [], str(exc)
        if not commands or any(not c.strip() for c in commands):
            evaluations.append({"commands": commands, "executable": False,
                                "flags": [], "blocked": True, "penalty": None,
                                "reason": parse_error or "no executable tool command"})
            continue
        per_command = [evaluate_candidate(cmd, history, problem_statement) for cmd in commands]
        evaluations.append({"commands": commands, "executable": True,
                            "flags": [flag for result in per_command for flag in result["flags"]],
                            "blocked": any(result["blocked"] for result in per_command),
                            "penalty": sum(result["penalty"] for result in per_command)})
    executable = [i for i, e in enumerate(evaluations) if e["executable"]]
    eligible = [i for i in executable if not evaluations[i]["blocked"]]
    if eligible:
        pool = eligible
    else:
        pool = executable
    if not pool:
        return {"chosen_idx": None, "chosen_command": None, "chosen_commands": [],
                "chosen_candidate": None, "requires_regeneration": True,
                "fallback_used": True, "all_candidates_blocked": True,
                "tie_break_seed": tie_break_seed, "tie_break_policy": "seeded_command_hash", "evaluations": evaluations}
    low = min(evaluations[i]["penalty"] for i in pool)
    ties = [i for i in pool if evaluations[i]["penalty"] == low]
    # Stable, seeded hash of the proposed commands avoids always choosing
    # candidate position zero. An identical repeated candidate is equivalent.
    def tie_key(i: int) -> tuple[str, int]:
        payload = json.dumps(evaluations[i]["commands"], sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(f"{tie_break_seed}|{payload}".encode("utf-8")).hexdigest()
        return digest, i
    chosen_idx = min(ties, key=tie_key)
    chosen_commands = evaluations[chosen_idx]["commands"]
    return {"chosen_idx": chosen_idx, "chosen_command": chosen_commands[0] if len(chosen_commands) == 1 else None,
            "chosen_commands": chosen_commands, "chosen_candidate": candidates[chosen_idx],
            "requires_regeneration": False, "fallback_used": not bool(eligible),
            "all_candidates_blocked": not bool(eligible), "tie_break_seed": tie_break_seed,
            "evaluations": evaluations}


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
    parser.add_argument("--tie-break-seed", type=int, default=0, help="Seed for unbiased tie-breaking")
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
        issue = issue_from_record(record)
        candidates = args.candidate

    result = select_candidate(candidates, history, issue, tie_break_seed=args.tie_break_seed)
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
