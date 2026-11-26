"""mini-swe-agent setup: the prompt fix, the model client, and the environment
that runs every command inside the run's container with a repeat guard."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from importlib.metadata import version as pkg_version

import litellm
import minisweagent
import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models.litellm_model import LitellmModel

from .config import Config
from .sandbox import PATHSPEC, Sandbox, sh

litellm.suppress_debug_info = True

REPO_NOTE = (
    "\n\nThe repository is checked out in /testbed, which is your working directory. "
    "Start by running `ls` to see the layout. Do not assume a src/ folder exists.\n"
)
# The default example made the model copy `find src ...` 29 times in our first run.
BAD_EXAMPLE = "find src -name '*.java' | grep -i builder"
GOOD_EXAMPLE = "find . -name '*.py' | grep -i builder"
DIFF_LIMIT = 20_000  # characters of the per-step diff kept in each step


def swebench_config() -> dict:
    """mini-swe-agent's own SWE-bench config file."""
    config_dir = os.path.join(os.path.dirname(minisweagent.__file__), "config")
    config_file = glob.glob(f"{config_dir}/**/swebench.yaml", recursive=True)[0]
    return yaml.safe_load(open(config_file))


def agent_config(cfg: Config) -> dict:
    """mini-swe-agent's SWE-bench config with our prompt fix and step limit."""
    agent_cfg = dict(swebench_config().get("agent", {}))
    agent_cfg["step_limit"] = cfg.step_limit
    for key, value in agent_cfg.items():
        if isinstance(value, str) and BAD_EXAMPLE in value:
            agent_cfg[key] = value.replace(BAD_EXAMPLE, GOOD_EXAMPLE)
    if "instance_template" not in agent_cfg:
        raise RuntimeError(f"mini-swe-agent config has no instance_template. Keys: {list(agent_cfg)}")
    agent_cfg["instance_template"] += REPO_NOTE
    return agent_cfg


class HistoryModel(LitellmModel):
    """Sends past assistant turns without their thinking text (Qwen's multi-turn advice), so prompts stay short
    and any engine (llama.cpp or vLLM) sees the same history. agent.messages keeps the thinking for the saved file."""

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        return [{k: v for k, v in m.items() if k != "reasoning_content"} if m.get("role") == "assistant" else m
                for m in prepared]


def make_model(cfg: Config, seed: int) -> LitellmModel:
    sampling = cfg.active_sampling
    top_k = sampling.pop("top_k", None)
    extra_body = {**({"top_k": top_k} if top_k is not None else {}), **cfg.chat_template_kwargs()}
    # The SWE-bench config's model templates: long outputs are cut to the first and last 5,000 characters,
    # and a response without a tool call gets guidance back. LitellmModel's defaults do neither.
    templates = {k: v for k, v in swebench_config().get("model", {}).items()
                 if k in ("observation_template", "format_error_template")}
    return HistoryModel(
        **templates,
        model_name=f"hosted_vllm/{cfg.model}",
        cost_tracking="ignore_errors",  # local models have no price; avoids a crash
        model_kwargs={"api_base": cfg.server.url, **sampling, "seed": seed, "extra_body": extra_body},
    )


def make_agent(cfg: Config, sandbox: Sandbox, seed: int) -> tuple["GuardedAgent", "SandboxEnvironment"]:
    env = SandboxEnvironment(sandbox, cfg.max_repeats)
    return GuardedAgent(make_model(cfg, seed), env, **agent_config(cfg)), env


class RepeatedFailure(InterruptAgentFlow):
    """Ends the run like mini-swe-agent's own limits do: with an `exit` message."""

    def __init__(self, command: str):
        super().__init__({"role": "exit", "content": "RepeatedFailure",
                          "extra": {"exit_status": "RepeatedFailure", "submission": "", "command": command}})


class GuardedAgent(DefaultAgent):
    """Stops after the repeat guard trips, but only once the last command's output is in messages[],
    so every tool call has its tool response (David's message-protocol check)."""

    def execute_actions(self, message: dict) -> list[dict]:
        observations = super().execute_actions(message)
        if self.env.stopped:
            raise RepeatedFailure(self.env.last_failed)
        return observations


class SandboxEnvironment(LocalEnvironment):
    """Runs every command inside the sandbox. Stops after `max_repeats` identical failing commands."""

    def __init__(self, sandbox: Sandbox, max_repeats: int, **kwargs):
        super().__init__(**kwargs)
        self.sandbox, self.max_repeats = sandbox, max_repeats
        self.last_failed, self.streak, self.stopped = None, 0, False

    def execute(self, action, *args, **kwargs):
        command = action["command"] if isinstance(action, dict) else action
        wrapped = self.sandbox.command(command)
        action = {**action, "command": wrapped} if isinstance(action, dict) else wrapped
        result = super().execute(action, *args, **kwargs)

        failed = isinstance(result, dict) and result.get("returncode", 0) != 0
        if failed and command == self.last_failed:
            self.streak += 1
        elif failed:
            self.last_failed, self.streak = command, 1
        else:
            self.last_failed, self.streak = None, 0

        if self.max_repeats and self.streak >= self.max_repeats:
            self.stopped = True  # GuardedAgent ends the run after this step's output is recorded
        if isinstance(result, dict):  # the code changes so far, saved in this step's tool message as extra.diff
            result.setdefault("extra", {})["diff"] = self.diff()
        return result

    def diff(self) -> str:
        """All changes since the starting snapshot. Uses its own git index file,
        so the agent's own `git status` and `git diff` output does not change."""
        base = self.sandbox.base
        out = self.sandbox.exec(
            'export GIT_INDEX_FILE="$(git rev-parse --absolute-git-dir)/goal_drift_index"; '
            f'[ -f "$GIT_INDEX_FILE" ] || git read-tree {base}; git add -A {PATHSPEC} 2>/dev/null; '
            f"git diff --cached {base}", timeout=300)
        return out if len(out) <= DIFF_LIMIT else out[:DIFF_LIMIT] + f"\n[diff cut at {DIFF_LIMIT} characters]"


def _chat_template(model: str) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return open(hf_hub_download(model, "chat_template.jinja")).read()
    except Exception:
        config = json.load(open(hf_hub_download(model, "tokenizer_config.json")))
        return str(config["chat_template"])


def get_versions(cfg: Config) -> dict[str, str]:
    """Package versions, the exact model revision and a chat template hash (all strings)."""
    info = {"backend": cfg.backend}
    for name in ["mini-swe-agent", "vllm", "udocker", "litellm"]:
        try:
            info[name] = pkg_version(name)
        except Exception:
            pass
    if cfg.backend == "apptainer":
        info[cfg.apptainer_bin] = sh(f"{cfg.apptainer_bin} --version")
    try:
        from huggingface_hub import model_info

        info["model_revision"] = model_info(cfg.model).sha
    except Exception as e:
        info["model_revision"] = f"unknown ({type(e).__name__})"
    try:
        info["chat_template_hash"] = hashlib.sha256(_chat_template(cfg.model).encode()).hexdigest()[:16]
    except Exception as e:
        info["chat_template_hash"] = f"unknown ({type(e).__name__})"
    return info
