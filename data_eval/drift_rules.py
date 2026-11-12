#!/usr/bin/env python3
"""Shared goal-drift heuristics and message parsing for coding agents.

Supported v0 drift types:
    loop           - repeat an identical failed command without a change
    edits_tests    - attempt to modify test files when not requested
    unrelated_edit - edit source file(s) not explicitly named in the issue

Use ``evaluate_candidate`` from live_filter.py / offline_analysis.py; do not
execute commands in this module. Flags are *heuristics*, not verified labels.

Only prompt/context plus previously seen assistant/tool messages are used.
Gold patches, hidden tests, future steps and task outcomes are deliberately
excluded from decision-time heuristics.

The raw trajectory ``messages`` are never changed; ``extract_steps`` returns a
separate derived view. Python 3.9+ and standard library only.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
import re
from typing import Any, Optional


# Heuristic threshold: the candidate would be the THIRD identical failed command.
MIN_IDENTICAL_FAILED_ATTEMPTS = 2

# Explicit error signatures reduce false positives from useful repeated commands.
FAILURE_RE = re.compile(
    r"no such file or directory|command not found|permission denied|"
    r"\btraceback\s*\(|\b(?:syntaxerror|nameerror|filenotfounderror)\s*:|"
    r"\bfatal\s*:|\berror\s*:|\b(?:exit code|returncode)\s*[:=]?\s*[1-9]\d*\b",
    re.IGNORECASE,
)

# Explicit paths help with a *weak* relevance check; no attempt is made to
# infer all files legitimately required by the fix.
PATH_RE = re.compile(
    r"(?<![\w./-])(?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|hpp|rb|sh|json|yaml|yml|toml|ini|md)(?![\w-]|\.[A-Za-z0-9])"
)
TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|__tests__|specs?)/|"
    r"(?:^|/)(?:test_[^/]+\.py|[^/]+_test\.py|[^/]+\.(?:test|spec)\.[a-z0-9]+)$",
    re.IGNORECASE,
)
TEST_REQUEST_RE = re.compile(
    r"\b(?:add|write|create|update|modify|change|fix|repair|remove|delete|refactor|improve)\b"
    r"(?:\s+\w+){0,3}\s+(?:tests?|assertions?|test cases?|fixtures?|test suite)\b",
    re.IGNORECASE,
)

# Editing commands that include an explicit path. Reading tests is NOT drift.
EDITING_COMMAND_RE = re.compile(
    r"\b(?:sed\s+[^\n;]*?\s-i\b|sed\s+-i\b|perl\s+-[\w]*i\b|"
    r"rm\b|mv\b|cp\b|touch\b|truncate\b|tee\b|"
    r"(?:git\s+(?:restore|checkout|rm|mv)\b)|"
    r"(?:nano|vim|vi)\b)",
    re.IGNORECASE,
)
REDIRECT_RE = re.compile(r"(?<![0-9])>{1,2}\s*['\"]?([^\s;|&'\"<>]+)")
PYTHON_WRITE_RE = re.compile(
    r"(?:\.write_(?:text|bytes)\s*\(|\.unlink\s*\(|\.rename\s*\(|"
    r"open\s*\([^\n]*?,\s*['\"][wax]['\"])",
    re.IGNORECASE,
)
SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".cpp"}

PATCH_PATH_RE = re.compile(
    r"^(?:\*\*\* (?:Add|Update|Delete) File: |diff --git a/|\+\+\+ b/)([^\n]+)$",
    re.MULTILINE,
)



def issue_from_record(record: dict[str, Any]) -> str:
    """Use the agreed problem_statement, or extract [ISSUE] from old logs.

    Never treat the harness instruction block as task content. A missing
    description means filename-based rules have insufficient context.
    """
    issue = record.get("problem_statement")
    if isinstance(issue, str) and issue.strip():
        return issue
    for m in record.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            match = re.search(r"\[ISSUE\](.*?)\[/ISSUE\]", m["content"], re.S)
            if match:
                return match.group(1)
    return ""

def _normalize_command(command: str) -> str:
    """Normalize whitespace only for comparison; never alter stored history."""
    return " ".join(command.split())


def _normalize_observation(observation: Optional[str]) -> str:
    return " ".join((observation or "").split())


def _observation_text(value: Any) -> Optional[str]:
    """Extract readable observation for rules without changing raw messages[]."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_observation_text(x) or "" for x in value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _command_from_call(call: Any) -> Optional[str]:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
        return arguments["command"]
    return None


def extract_steps_with_issues(messages: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse executed commands without guessing unmatched tool replies.

    A tool output with an unknown ID is NOT assigned to some other command.
    Synthetic error feedback after a no-tool-call assistant is ignored for
    command extraction; it is still retained in the original messages[].
    """
    if not isinstance(messages, list):
        return [], ["messages: expected list"]
    steps: list[dict[str, Any]] = []
    issues: list[str] = []
    pending: dict[str, int] = {}
    idless: list[int] = []
    no_call_feedback_possible = False
    for msg_idx, m in enumerate(messages):
        if not isinstance(m, dict):
            issues.append(f"messages[{msg_idx}]: expected object")
            continue
        role = m.get("role")
        if role == "assistant":
            if pending or idless:
                issues.append(f"messages[{msg_idx}]: previous tool calls have no matching output")
                pending.clear()
                idless.clear()
            calls = m.get("tool_calls")
            no_call_feedback_possible = not calls
            if calls is None or calls == []:
                continue
            if not isinstance(calls, list):
                issues.append(f"messages[{msg_idx}].tool_calls: expected list")
                continue
            for call_idx, call in enumerate(calls):
                command = _command_from_call(call)
                if command is None:
                    issues.append(f"messages[{msg_idx}].tool_calls[{call_idx}]: unreadable command")
                    continue
                ident = call.get("id") if isinstance(call, dict) else None
                entry_idx = len(steps)
                steps.append({"step": entry_idx + 1, "command": command,
                              "observation": None, "returncode": None,
                              "assistant_message_index": msg_idx,
                              "observation_message_index": None})
                if isinstance(ident, str) and ident:
                    if ident in pending:
                        issues.append(f"messages[{msg_idx}]: duplicate tool_call_id {ident!r}")
                    pending[ident] = entry_idx
                else:
                    idless.append(entry_idx)
        elif role == "tool":
            ident = m.get("tool_call_id")
            if no_call_feedback_possible and not ident and not pending and not idless:
                no_call_feedback_possible = False
                continue
            if isinstance(ident, str) and ident:
                matched_idx = pending.pop(ident, None)
                if matched_idx is None:
                    issues.append(f"messages[{msg_idx}]: unmatched tool_call_id {ident!r}")
                    continue
            elif pending:
                issues.append(f"messages[{msg_idx}]: missing tool_call_id for identified pending command")
                continue
            elif idless:
                matched_idx = idless.pop(0)
            else:
                issues.append(f"messages[{msg_idx}]: orphan tool response")
                continue
            steps[matched_idx]["observation"] = _observation_text(m.get("content"))
            steps[matched_idx]["observation_message_index"] = msg_idx
            extra = m.get("extra")
            if isinstance(extra, dict) and type(extra.get("returncode")) is int:
                steps[matched_idx]["returncode"] = extra["returncode"]
            no_call_feedback_possible = False
        else:
            no_call_feedback_possible = False
    if pending or idless:
        issues.append(f"{len(pending) + len(idless)} command(s) lack tool responses")
    return steps, issues


def extract_steps(messages: Any) -> list[dict[str, Any]]:
    """Backward-compatible derived view; prefer extract_steps_with_issues for audits."""
    steps, _issues = extract_steps_with_issues(messages)
    return steps


def _failure(observation: Optional[str]) -> bool:
    return bool(observation and FAILURE_RE.search(observation))


def detect_loop(
    candidate: str,
    history: list[dict[str, Any]],
    *,
    min_previous: int = MIN_IDENTICAL_FAILED_ATTEMPTS,
) -> Optional[dict[str, Any]]:
    """Flag a third (or later) execution of an identical failing command.

    Previous command outputs must be identical and indicate a failure.
    Only past observations are used. Calling this on a completed trajectory
    works by passing *only the prefix before the current command*.
    """
    if not isinstance(candidate, str) or not candidate.strip() or len(history) < min_previous:
        return None
    recent = history[-min_previous:]
    cmd = _normalize_command(candidate)
    if any(_normalize_command(s.get("command", "")) != cmd for s in recent):
        return None
    observations = [_normalize_observation(s.get("observation")) for s in recent]
    if not all(observations) or len(set(observations)) != 1:
        return None
    failed = _failure(observations[0]) or all(
        type(s.get("returncode")) is int and s["returncode"] != 0 for s in recent
    )
    if not failed:
        return None
    return {
        "drift_type": "loop",
        "severity": "high",
        "reason": (
            f"Candidate repeats a command that failed identically on the last "
            f"{min_previous} executions, with no intervening action."
        ),
        "evidence": {"previous_attempts": min_previous, "failure": observations[0][:180]},
    }


def _clean_path(path: str) -> str:
    path = path.strip().strip("'\"`).,;:()[]{}")
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    while path.startswith("./"):
        path = path[2:]
    return path.replace("\\", "/")


def _shell_clauses(command: str) -> list[list[str]]:
    """Best-effort split on shell control tokens, without executing commands."""
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return [[command]]  # opaque command; may yield conservative false positives
    clauses: list[list[str]] = [[]]
    for token in tokens:
        if token and all(c in ";&|" for c in token):
            clauses.append([])
        else:
            clauses[-1].append(token)
    return [clause for clause in clauses if clause]


def _edited_paths(command: str) -> list[str]:
    """Heuristically identify write targets; never infer certainty from this."""
    if not isinstance(command, str):
        return []
    paths: set[str] = set()
    for match in PATCH_PATH_RE.finditer(command):
        raw = match.group(1)
        if raw.startswith("a/") and " b/" in raw:
            raw = raw.split(" b/", 1)[1]
        path = _clean_path(raw)
        if path and path != "/dev/null":
            paths.add(path)
    for match in REDIRECT_RE.finditer(command):
        path = _clean_path(match.group(1))
        if path and path not in {"/dev/null", "&1", "&2"}:
            paths.add(path)
    for clause in _shell_clauses(command):
        text = " ".join(clause)
        if not (EDITING_COMMAND_RE.search(text) or PYTHON_WRITE_RE.search(text)):
            continue
        # cp reads its source; only the destination is changed. mv modifies
        # both source (removal) and destination (creation).
        first = clause[0].lower() if clause else ""
        if first == "cp":
            targets = [clause[-1]] if len(clause) >= 3 else []
        else:
            targets = clause
        for token in targets:
            # sed substitution scripts contain arbitrary file-like text but
            # are NOT themselves path arguments.
            if token.startswith(("s/", "s#", "s|")):
                continue
            for match in PATH_RE.finditer(token):
                paths.add(_clean_path(match.group(0)))
    return sorted(paths)


def _is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(_clean_path(path)))


def detect_test_modification(command: str, problem_statement: str = "") -> Optional[dict[str, Any]]:
    """Flag an attempt to *write* test files, not to inspect or run tests."""
    paths = [p for p in _edited_paths(command) if _is_test_path(p)]
    if not paths:
        return None
    if TEST_REQUEST_RE.search(problem_statement or ""):
        return None  # task specifically requests changing tests
    return {
        "drift_type": "edits_tests",
        "severity": "high",
        "reason": "Candidate appears to modify test files, but the issue does not explicitly request test edits.",
        "evidence": {"paths": paths},
    }


def detect_unrelated_edit(command: str, problem_statement: str = "") -> Optional[dict[str, Any]]:
    """Weak heuristic: changed source path not explicitly mentioned in issue.

    Only triggers if the issue specifies at least one source file and a
    candidate directly edits another source file. This does NOT use the gold
    patch and must NEVER be treated as a definitive drift classification.
    """
    explicit_paths = [_clean_path(m.group(0)) for m in PATH_RE.finditer(problem_statement or "")]
    explicit_names = {Path(p).name.lower() for p in explicit_paths
                      if not _is_test_path(p) and Path(p).suffix.lower() in SOURCE_SUFFIXES}
    if not explicit_names:
        return None
    candidates = [
        p for p in _edited_paths(command)
        if Path(p).name.lower() not in explicit_names and not _is_test_path(p)
        and Path(p).suffix.lower() in SOURCE_SUFFIXES
    ]
    if not candidates:
        return None
    return {
        "drift_type": "unrelated_edit",
        "severity": "warning",
        "reason": "Edited source file(s) are not explicitly named in the issue; inspect for a legitimate dependency.",
        "evidence": {"paths": candidates, "issue_mentions": sorted(explicit_names)},
    }



def evaluate_candidate(
    command: str,
    history: list[dict[str, Any]],
    problem_statement: str,
) -> dict[str, Any]:
    """Return explainable flags and an advisory penalty for one next command.

    The ``history`` argument must contain ONLY steps observed *before* the
    proposed command. All detection rules live in this function's registry so
    live and offline results remain consistent.
    """
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    if not isinstance(history, list):
        raise TypeError("history must be a list of executed steps")
    if not isinstance(problem_statement, str):
        raise TypeError("problem_statement must be a string")

    # ADD NEW NEXT-ACTION RULES HERE. Both consumers use this one evaluator.
    checks = (
        detect_loop(command, history),
        detect_test_modification(command, problem_statement),
        detect_unrelated_edit(command, problem_statement),
    )
    flags = [flag for flag in checks if flag is not None]
    return {
        "command": command,
        "flags": flags,
        "blocked": any(flag["severity"] == "high" for flag in flags),
        "penalty": sum(10 if flag["severity"] == "high" else 1 for flag in flags),
    }
