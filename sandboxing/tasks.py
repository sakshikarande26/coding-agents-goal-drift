"""R2E-Gym tasks: loading, the answer key, and the gold fix.

The gold fix and the answer key are used only for the smoke test and for grading.
The agent never sees them.
"""

from __future__ import annotations

import datetime
import json
import random
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


FILTER_COLUMNS = ["commit_hash", "repo_name", "num_non_test_files", "num_non_test_lines"]


def _dataset(cfg: Config):
    """The whole split, downloaded once into the Hugging Face cache (about 1 GB for R2E-Gym-Lite train),
    so parallel jobs read it from disk instead of each streaming it."""
    from datasets import load_dataset

    return load_dataset(cfg.dataset, split=cfg.dataset_split)


def _small_fix(row: dict, cfg: Config) -> bool:
    return row["num_non_test_files"] <= cfg.max_non_test_files and row["num_non_test_lines"] <= cfg.max_non_test_lines


def _round_robin(tasks: list[dict]) -> list[dict]:
    """Interleave repos (A1 B1 C1 ... A2 B2 C2 ...), so the first k tasks of a split cover the repos evenly."""
    by_repo: dict[str, list[dict]] = {}
    for t in tasks:
        by_repo.setdefault(t["repo"], []).append(t)
    queues = [by_repo[r] for r in sorted(by_repo)]
    out = []
    for i in range(max(map(len, queues), default=0)):
        out += [q[i] for q in queues if i < len(q)]
    return out


def sample_task_list(cfg: Config) -> dict:
    """Seeded random sample of small-fix tasks, the same number from every repo,
    split into train and held-out inside each repo. Same settings = same list."""
    pool: dict[str, list[str]] = {}
    for row in _dataset(cfg).select_columns(FILTER_COLUMNS):
        if _small_fix(row, cfg):
            pool.setdefault(row["repo_name"], []).append(row["commit_hash"])
    need = cfg.train_per_repo + cfg.heldout_per_repo
    short = {repo: len(ids) for repo, ids in pool.items() if len(ids) < need}
    if short:
        raise ValueError(f"These repos have fewer than {need} small-fix tasks: {short}. Lower the per-repo counts.")
    rng = random.Random(cfg.fixed_seed)
    train, heldout = [], []
    for repo in sorted(pool):
        picked = rng.sample(sorted(pool[repo]), need)  # sorted first, so dataset order does not matter
        train += [{"task_id": t, "repo": repo, "split": "train"} for t in picked[:cfg.train_per_repo]]
        heldout += [{"task_id": t, "repo": repo, "split": "heldout"} for t in picked[cfg.train_per_repo:]]
    return {
        "dataset": cfg.dataset,
        "dataset_split": cfg.dataset_split,
        "filter": {"max_non_test_files": cfg.max_non_test_files, "max_non_test_lines": cfg.max_non_test_lines},
        "fixed_seed": cfg.fixed_seed,
        "train_per_repo": cfg.train_per_repo,
        "heldout_per_repo": cfg.heldout_per_repo,
        "pool_per_repo": {repo: len(ids) for repo, ids in sorted(pool.items())},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tasks": _round_robin(train) + _round_robin(heldout),
    }


def save_task_list(cfg: Config, task_list: dict) -> None:
    """The list is frozen once made: train and held-out tasks must never change. Refuse to overwrite."""
    if cfg.task_list.exists():
        raise FileExistsError(f"{cfg.task_list} already exists. The task list is frozen; delete it by hand only "
                              "if no runs were made from it.")
    cfg.task_list.parent.mkdir(parents=True, exist_ok=True)
    cfg.task_list.write_text(json.dumps(task_list, indent=1) + "\n")


def load_tasks(cfg: Config, shard: tuple[int, int] = (0, 1)) -> list[Task]:
    """Tasks of cfg.split from the frozen task list, in list order (repos interleaved).
    n_tasks keeps the first n (e.g. a pilot). shard (i, n) keeps every n-th task from i, for parallel jobs."""
    if not cfg.task_list.exists():
        raise FileNotFoundError(f"No task list at {cfg.task_list}. Run `goal-drift sample` once first.")
    task_list = json.loads(cfg.task_list.read_text())
    if task_list["dataset"] != cfg.dataset or task_list["dataset_split"] != cfg.dataset_split:
        raise ValueError(f"{cfg.task_list} was made from {task_list['dataset']}:{task_list['dataset_split']}, "
                         f"but the config says {cfg.dataset}:{cfg.dataset_split}")
    ids = [t["task_id"] for t in task_list["tasks"] if t["split"] == cfg.split]
    if not ids:
        raise ValueError(f"{cfg.task_list} has no {cfg.split!r} tasks")
    ids = ids[:cfg.n_tasks] if cfg.n_tasks else ids
    index, count = shard
    ids = ids[index::count]
    ds = _dataset(cfg)
    position = {task_id: i for i, task_id in enumerate(ds["commit_hash"])}
    missing = [i for i in ids if i not in position]
    if missing:
        raise ValueError(f"{len(missing)} listed task(s) not found in {cfg.dataset}: {missing[:3]}")
    return [Task(ds[position[i]]) for i in ids]
