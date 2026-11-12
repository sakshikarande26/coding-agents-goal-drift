"""Containers for R2E-Gym task images.

Backends: apptainer (Unity), docker (any machine with Docker), udocker (Colab, COOK).
`backend: auto` in the config picks one (see config.detect_backend).

Every run gets its own container name and its own folder under <work_dir>/runs/<run_id>/,
so several runs can go at the same time without overwriting each other.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import Config

AGENT_ENV = "export GIT_PAGER=cat PAGER=cat; "  # git never waits for a keypress

# Paths that never count as agent edits.
PATHSPEC = (
    "-- . ':(exclude).venv' ':(exclude)r2e_tests' ':(exclude).pytest_cache' "
    "':(exclude)patch.txt' ':(glob,exclude)**/__pycache__/**' ':(glob,exclude)**/*.pyc'"
)

# Hidden-test folders, relative to the container's / .
TEST_DIRS = ["r2e_tests", "testbed/r2e_tests"]


def sh(command: str, timeout: int | None = None) -> str:
    """Run a host shell command and return stdout + stderr."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


class Sandbox:
    """One writable container for one run."""

    def __init__(self, cfg: Config, image: str, run_id: str):
        self.cfg = cfg
        self.image = image
        self.run_id = run_id
        self.name = "r2e_" + safe_name(run_id)
        self.run_dir = cfg.runs_dir / safe_name(run_id)
        self.hidden_dir = self.run_dir / "hidden_tests"
        self.root: Path | None = None  # container filesystem on the host (not used by docker)
        self.base: str | None = None   # git tree hash of the starting code

    # --- backend details ---

    @property
    def backend(self) -> str:
        return self.cfg.backend

    @property
    def _udocker(self) -> str:
        # Colab runs as root and needs --allow-root; clusters do not run as root.
        return "udocker --allow-root" if os.geteuid() == 0 else "udocker"

    @property
    def _sif(self) -> Path:
        return self.cfg.sif_dir / (safe_name(self.image) + ".sif")

    @property
    def _pristine(self) -> Path:
        """Unpacked image, built once per image and shared by all runs (apptainer)."""
        return self.cfg.pristine_dir / safe_name(self.image)

    def command(self, command: str, workdir: str = "/testbed") -> str:
        """The host command that runs `command` inside this container."""
        inner = shlex.quote(AGENT_ENV + command)
        if self.backend == "udocker":
            return f"{self._udocker} run --nobanner --workdir={workdir} {self.name} /bin/bash -lc {inner}"
        if self.backend == "docker":
            return f"docker exec -w {workdir} -e GIT_PAGER=cat {self.name} /bin/bash -lc {inner}"
        # --silent: no chdir warnings in the model's observations
        # --containall: no host $HOME, /tmp or environment variables (HF_TOKEN etc.)
        return (
            f"{self.cfg.apptainer_bin} --silent exec --containall --writable --pwd {workdir} "
            f"--env GIT_PAGER=cat {self.root} /bin/bash -lc {inner}"
        )

    def exec(self, command: str, workdir: str = "/testbed", timeout: int = 900) -> str:
        return sh(self.command(command, workdir), timeout=timeout)

    def _exists(self, path_in_box: str) -> bool:
        if self.backend == "docker":
            return sh(f"docker exec {self.name} test -d /{path_in_box} && echo yes") == "yes"
        return (self.root / path_in_box).is_dir() and not (self.root / path_in_box).is_symlink()

    # --- lifecycle ---

    def pull(self) -> str:
        if self.backend == "udocker":
            return sh(f"{self._udocker} pull {self.image}")[-300:]
        if self.backend == "docker":
            return sh(f"docker pull -q {self.image}")[-300:]
        if self._sif.exists():
            return f"SIF already there: {self._sif}"
        self.cfg.sif_dir.mkdir(parents=True, exist_ok=True)
        return sh(f"{self.cfg.apptainer_bin} pull {self._sif} docker://{self.image}")[-300:]

    def fresh(self) -> None:
        """New writable container with the buggy code, then snapshot its working tree as `base`."""
        self.cleanup()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "udocker":
            sh(f"{self._udocker} create --name={self.name} {self.image}")
            self.root = Path(sh(f"{self._udocker} inspect -p {self.name}").splitlines()[-1])
        elif self.backend == "docker":
            out = sh(f"docker run -d --name {self.name} -w /testbed {self.image} sleep infinity")
            if not self._exists("testbed"):
                raise RuntimeError(f"Docker container has no /testbed: {out[-500:]}")
        else:
            if not self._pristine.is_dir():
                # Build under a temp name, then rename, so parallel runs never see a half-built copy.
                tmp = self._pristine.with_name(self._pristine.name + f".tmp{os.getpid()}")
                self.cfg.pristine_dir.mkdir(parents=True, exist_ok=True)
                out = sh(f"{self.cfg.apptainer_bin} build --sandbox {tmp} {self._sif}")
                if not tmp.is_dir():
                    raise RuntimeError(f"Could not build sandbox from {self._sif}: {out[-500:]}")
                try:
                    tmp.rename(self._pristine)
                except OSError:
                    shutil.rmtree(tmp, ignore_errors=True)  # another run finished first
            self.root = self.run_dir / "root"
            sh(f"cp -a {self._pristine} {self.root}")
        if self.backend != "docker" and not (self.root and (self.root / "testbed").is_dir()):
            raise RuntimeError(f"Container root has no /testbed: {self.root}")
        self.base = self.exec(f"git add -A {PATHSPEC} && git write-tree").splitlines()[-1].strip()
        if not re.fullmatch(r"[0-9a-f]{40}", self.base):
            raise RuntimeError(f"Could not snapshot the working tree: {self.base}")

    def write_file(self, path: str, content: str) -> None:
        """Write a file under /testbed from the host (used for the gold fix)."""
        if self.backend == "docker":
            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                f.write(content)
            sh(f"docker cp {f.name} {self.name}:/testbed/{path}")
            os.unlink(f.name)
            return
        target = self.root / "testbed" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def hide_tests(self) -> None:
        """Move the hidden tests out of the container while the agent works."""
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        for test_dir in TEST_DIRS:
            if not self._exists(test_dir):
                continue
            outside = self.hidden_dir / test_dir.replace("/", "__")
            shutil.rmtree(outside, ignore_errors=True)
            if self.backend == "docker":
                sh(f"docker cp {self.name}:/{test_dir} {outside} && docker exec {self.name} rm -rf /{test_dir}")
            else:
                shutil.move(str(self.root / test_dir), str(outside))

    def restore_tests(self) -> None:
        for test_dir in TEST_DIRS:
            outside = self.hidden_dir / test_dir.replace("/", "__")
            if not outside.exists():
                continue
            if self.backend == "docker":
                sh(f"docker cp {outside} {self.name}:/{test_dir}")
                shutil.rmtree(outside, ignore_errors=True)
            else:
                shutil.move(str(outside), str(self.root / test_dir))

    def cleanup(self) -> None:
        """Remove this run's container and folder. Keeps the shared image."""
        if self.backend == "udocker":
            sh(f"{self._udocker} rm {self.name}")
        elif self.backend == "docker":
            sh(f"docker rm -f {self.name}")
        if self.run_dir.exists():
            sh(f"chmod -R u+w {shlex.quote(str(self.run_dir))}")  # apptainer copies can be read-only
            shutil.rmtree(self.run_dir, ignore_errors=True)
        self.root, self.base = None, None

    def remove_image(self) -> None:
        """Free disk after a task. Apptainer keeps the SIF, since pulls are slow."""
        if self.backend == "udocker":
            sh(f"{self._udocker} rmi {self.image}")
        elif self.backend == "docker":
            sh(f"docker rmi {self.image}")
        elif self._pristine.exists():
            sh(f"chmod -R u+w {shlex.quote(str(self._pristine))}")
            shutil.rmtree(self._pristine, ignore_errors=True)
