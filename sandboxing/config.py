"""Load the YAML settings file and check every value before anything runs."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# David's validator is the contract: take the allowed values from it, never copy them.
from data_eval.validate_trajectories import CONDITIONS, SPLITS

BACKENDS = {"apptainer", "docker", "udocker"}
SAMPLING_KEYS = {"temperature", "top_p", "presence_penalty"}  # required by David's validator

# Qwen3 models that have no thinking mode (they must run with thinking: false).
NO_THINKING_MARKERS = ("Instruct-2507",)


class ConfigError(ValueError):
    pass


def docker_works() -> bool:
    """True only if the docker command exists and its daemon answers."""
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_backend(apptainer_bin: str = "apptainer") -> str:
    """apptainer if installed (Unity), else docker if it runs (your own machine), else udocker
    (Colab, COOK). COOK's `singularity` is skipped on purpose: it cannot load Docker images there."""
    if shutil.which(apptainer_bin):
        return "apptainer"
    if docker_works():
        return "docker"
    return "udocker"


@dataclass
class ServerConfig:
    start: bool = True
    url: str = "http://localhost:8000/v1"
    dtype: str = "half"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.85
    startup_timeout: int = 900


@dataclass
class Config:
    backend: str
    work_dir: Path
    model: str
    thinking: bool
    sampling: dict
    server: ServerConfig = field(default_factory=ServerConfig)
    apptainer_bin: str = "apptainer"
    cleanup_images: bool = True
    dataset: str = "R2E-Gym/R2E-Gym-Lite"
    dataset_split: str = "train"
    n_tasks: int = 5
    max_per_repo: int = 2
    max_non_test_files: int = 1
    max_non_test_lines: int = 10
    seeds: list = field(default_factory=lambda: [0])
    step_limit: int = 30
    max_repeats: int = 5
    condition: str = "unguided"
    split: str = "train"

    # --- derived values ---

    @property
    def model_has_thinking(self) -> bool:
        return not any(marker in self.model for marker in NO_THINKING_MARKERS)

    @property
    def active_sampling(self) -> dict:
        """The sampling preset that matches the thinking switch."""
        return dict(self.sampling["thinking_on" if self.thinking else "thinking_off"])

    def chat_template_kwargs(self) -> dict:
        """Extra request body that turns Qwen3 thinking on or off."""
        if not self.model_has_thinking:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": self.thinking}}

    @property
    def trajs_dir(self) -> Path:
        return self.work_dir / "trajs"

    @property
    def runs_dir(self) -> Path:
        """One folder per run: container copy (apptainer) and hidden tests."""
        return self.work_dir / "runs"

    @property
    def sif_dir(self) -> Path:
        return self.work_dir / "sifs"

    @property
    def pristine_dir(self) -> Path:
        return self.work_dir / "pristine"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"


def _check_keys(data: dict, cls: type, where: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown setting(s) {sorted(unknown)}. Known: {sorted(known)}")


def _validate(cfg: Config) -> None:
    problems = []
    if cfg.backend not in BACKENDS:
        problems.append(f"backend must be auto or one of {sorted(BACKENDS)}, got {cfg.backend!r}")
    if cfg.backend == "apptainer" and cfg.work_dir.is_relative_to(Path.home()):
        # Khoa's Unity rule: hidden tests, gold patches and trajectories live outside $HOME.
        problems.append(f"work_dir must be outside $HOME with apptainer (e.g. /work/pi_<group>/...), got {cfg.work_dir}")
    if cfg.split not in SPLITS:
        problems.append(f"split must be one of {sorted(SPLITS)}, got {cfg.split!r}")
    if cfg.condition not in CONDITIONS:
        problems.append(f"condition must be one of {sorted(CONDITIONS)}, got {cfg.condition!r}")
    if not isinstance(cfg.thinking, bool):
        problems.append("thinking must be true or false")
    elif cfg.thinking and not cfg.model_has_thinking:
        problems.append(f"thinking: true needs a hybrid model such as Qwen/Qwen3-4B; {cfg.model} has no thinking mode")
    for preset in ("thinking_on", "thinking_off"):
        values = cfg.sampling.get(preset) if isinstance(cfg.sampling, dict) else None
        if not isinstance(values, dict):
            problems.append(f"sampling.{preset} is missing")
        elif SAMPLING_KEYS - set(values):
            problems.append(f"sampling.{preset} needs {sorted(SAMPLING_KEYS - set(values))}")
    if not cfg.seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in cfg.seeds):
        problems.append("seeds must be a non-empty list of integers")
    if cfg.step_limit < 1:
        problems.append("step_limit must be >= 1")
    if cfg.max_repeats < 0:
        problems.append("max_repeats must be >= 0 (0 = off)")
    if cfg.n_tasks < 1:
        problems.append("n_tasks must be >= 1")
    if problems:
        raise ConfigError("Bad config:\n  - " + "\n  - ".join(problems))


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Read a YAML file, apply command-line overrides, and check the result."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    _check_keys(data, Config, str(path))
    server = data.pop("server", {}) or {}
    _check_keys(server, ServerConfig, f"{path}: server")
    data["work_dir"] = Path(data.get("work_dir", "~/goal-drift-work")).expanduser().resolve()
    if data.get("backend", "auto") == "auto":
        data["backend"] = detect_backend(data.get("apptainer_bin", "apptainer"))
    cfg = Config(server=ServerConfig(**server), **data)
    _validate(cfg)
    return cfg
