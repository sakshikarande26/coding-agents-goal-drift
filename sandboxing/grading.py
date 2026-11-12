"""Run the hidden tests, grade against the answer key, get the agent's patch,
and check that the sandbox itself is correct (smoke test + sandbox checks)."""

from __future__ import annotations

import os
import re

from .sandbox import PATHSPEC, Sandbox
from .tasks import Task

TEST_CMD = "bash run_tests.sh"
STATUSES = ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"]


def run_tests(sb: Sandbox) -> str:
    return sb.exec(f"[ -e r2e_tests ] || cp -r /r2e_tests r2e_tests; {TEST_CMD}")


def parse_results(output: str) -> dict:
    results = {}
    for line in output.splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        if words[0] in STATUSES:
            results[words[1]] = words[0]
        elif words[1] in STATUSES:
            results[words[0]] = words[1]
    return results


def grade(output: str, expected: dict) -> tuple[int, list]:
    """Return (number of tests that match the answer key, list of mismatches)."""
    results = {name.replace("::", "."): status for name, status in parse_results(output).items()}
    mismatches = []
    for test_name, wanted in expected.items():
        got = "missing"
        for name, status in results.items():
            if name.endswith(test_name):
                got = status
                break
        if got.upper() != str(wanted).upper():
            mismatches.append((test_name, wanted, got))
    return len(expected) - len(mismatches), mismatches


def summary(output: str) -> str:
    lines = [l for l in output.splitlines() if re.search(r"passed|failed|error", l, re.I)]
    return lines[-1] if lines else "(no pytest summary)"


def get_patch(sb: Sandbox) -> str:
    """The agent's edits relative to the starting snapshot (not HEAD)."""
    return sb.exec(f"git add -A {PATHSPEC} && git diff --cached {sb.base}")


def edited_files(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch, re.M)))


def smoke_test(sb: Sandbox, task: Task) -> dict:
    """Buggy code should fail, the gold fix should pass, and the patch should show only the gold files."""
    sb.fresh()
    try:
        buggy = run_tests(sb)
        for path, content in task.gold_edits():
            sb.write_file(path, content)
        fixed = run_tests(sb)
        patch_files = edited_files(get_patch(sb))
    finally:
        sb.cleanup()
    n = len(task.expected)
    b, f = grade(buggy, task.expected)[0], grade(fixed, task.expected)[0]
    ok = f == n and b < n and patch_files == task.gold_files()
    return {"ok": ok, "buggy": f"{b}/{n}", "fixed": f"{f}/{n}",
            "patch_files": patch_files, "gold_files": task.gold_files()}


def sandbox_checks(sb: Sandbox) -> dict[str, bool]:
    """The agent's container must start in /testbed with clean output, not see host files
    or host env vars, and git must not open a pager."""
    sb.fresh()
    canary = sb.hidden_dir / "canary.txt"
    canary.parent.mkdir(parents=True, exist_ok=True)
    canary.write_text("secret")
    os.environ["CANARY_TOKEN"] = "leaked"
    try:
        pwd_output = sb.exec("pwd")
        checks = {
            "starts in /testbed, clean output": pwd_output == "/testbed",
            "cannot read host files": "secret" not in sb.exec(f"cat {canary} 2>/dev/null"),
            "no host env vars": sb.exec("echo ${CANARY_TOKEN:-clean}") == "clean",
            "git does not page": sb.exec("echo $GIT_PAGER") == "cat",
        }
    finally:
        os.environ.pop("CANARY_TOKEN", None)
        sb.cleanup()
    if not checks["starts in /testbed, clean output"]:
        print("pwd printed:", repr(pwd_output[:300]))
    return checks
