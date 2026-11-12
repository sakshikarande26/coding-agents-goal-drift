"""R2E-Gym tasks: loading, the answer key, and the gold fix.

The gold fix and the answer key are used only for the smoke test and for grading.
The agent never sees them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

from .config import Config

TEST_DIR_NAMES = {"tests", "test", "testing", "r2e_tests"}


def is_test_path(path: str) -> bool:
    """True if the file sits inside a test folder (not just any path containing 'test')."""
    return any(part in TEST_DIR_NAMES for part in path.split("/")[:-1])


def file_path(file_diff: dict) -> str | None:
    return ((file_diff.get("header") or {}).get("file") or {}).get("path")


@dataclass
class Task:
    row: dict

    @property
    def task_id(self) -> str:
        return self.row["commit_hash"]

    @property
    def repo(self) -> str:
        return self.row["repo_name"]

    @property
    def image(self) -> str:
        return self.row["docker_image"]

    @property
    def problem_statement(self) -> str:
        return self.row["problem_statement"]

    @cached_property
    def expected(self) -> dict:
        """Answer key: {test name: status after the fix}."""
        return json.loads(self.row["expected_output_json"])

    @cached_property
    def commit(self) -> dict:
        """Gold fix, parsed."""
        return json.loads(self.row["parsed_commit_content"])

    def gold_files(self) -> list[str]:
        paths = (file_path(d) for d in self.commit["file_diffs"])
        return sorted(p for p in paths if p and not is_test_path(p))

    def gold_edits(self) -> list[tuple[str, str]]:
        """(path, new file content) for every non-test file in the gold fix."""
        edits = []
        for file_diff in self.commit["file_diffs"]:
            path, content = file_path(file_diff), file_diff.get("new_file_content")
            if path and content is not None and not is_test_path(path):
                edits.append((path, content))
        return edits


def _stream(cfg: Config):
    from datasets import load_dataset

    return load_dataset(cfg.dataset, split=cfg.dataset_split, streaming=True)


def load_tasks(cfg: Config) -> list[Task]:
    """Small fixes only, with a cap per repo so we keep some variety."""
    tasks, per_repo = [], {}
    for row in _stream(cfg):
        if row["num_non_test_files"] > cfg.max_non_test_files or row["num_non_test_lines"] > cfg.max_non_test_lines:
            continue
        repo = row["repo_name"]
        if per_repo.get(repo, 0) >= cfg.max_per_repo:
            continue
        per_repo[repo] = per_repo.get(repo, 0) + 1
        tasks.append(Task(row))
        if len(tasks) >= cfg.n_tasks:
            break
    return tasks
