#!/usr/bin/env python3
"""Measure issue-filename coverage for the existing unrelated_edit heuristic.

Run from the repository root:
  pip install datasets
  python -m data_eval.check_filename_coverage
  python -m data_eval.check_filename_coverage --task-ids selected_task_ids.txt

The optional IDs file contains one R2E-Gym commit_hash per line.
The script only reads problem_statement; reference/gold patches are never used.
"""

import argparse
import csv
from pathlib import Path

try:
    from .drift_rules import PATH_RE, _clean_path, _is_test_path, SOURCE_SUFFIXES
except ImportError:
    from drift_rules import PATH_RE, _clean_path, _is_test_path, SOURCE_SUFFIXES


def paths_mentioned(issue: str) -> tuple[list[str], list[str]]:
    """Match the existing detector's source-mention gate and stricter source-file subset."""
    all_paths = sorted({
        _clean_path(m.group(0))
        for m in PATH_RE.finditer(issue or "")
    })
    non_test = [p for p in all_paths if not _is_test_path(p)]
    sources = [p for p in non_test if Path(p).suffix.lower() in SOURCE_SUFFIXES]
    return non_test, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=Path, help="Text file with one selected commit_hash per line")
    parser.add_argument("--output", type=Path, default=Path("filename_coverage.csv"))
    args = parser.parse_args()

    ids = None
    if args.task_ids:
        ids = {line.strip() for line in args.task_ids.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")}
        if not ids:
            parser.error("Task ID file is empty")

    # Mirrors Sakshi's notebook; streaming avoids downloading the full dataset up front.
    from datasets import load_dataset
    dataset = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train", streaming=True)
    results = []
    for task in dataset:
        task_id = task["commit_hash"]
        if ids is not None and task_id not in ids:
            continue
        non_test, sources = paths_mentioned(task.get("problem_statement") or "")
        results.append({
            "task_id": task_id,
            "repo": task["repo_name"],
            "has_any_non_test_filename": bool(non_test),
            "has_source_filename": bool(sources),
            "non_test_filenames": "; ".join(non_test),
            "source_filenames": "; ".join(sources),
        })
        if ids is not None and len(results) == len(ids):
            break

    if not results:
        parser.error("No matching tasks found in R2E-Gym-Lite training split")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    any_count = sum(r["has_any_non_test_filename"] for r in results)
    source_count = sum(r["has_source_filename"] for r in results)
    print(f"Tasks analyzed: {total}" + (" (selected task IDs)" if ids else " (full training split)"))
    print(f"Any non-test filename mentioned: {any_count}/{total} ({any_count/total:.1%})")
    print(f"Actual current unrelated_edit source-filename gate: {source_count}/{total} ({source_count/total:.1%})")
    print(f"Explicit source-code filename: {source_count}/{total} ({source_count/total:.1%})")
    print("Note: filename gate is only eligibility; triggering also requires a detectable edit command.")
    if ids is not None and total != len(ids):
        print(f"WARNING: {len(ids) - total} selected ID(s) not found in training split")
    print(f"Details: {args.output}")


if __name__ == "__main__":
    main()
