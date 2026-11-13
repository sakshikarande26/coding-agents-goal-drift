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
from .sandbox import Sandbox, sh

litellm.suppress_debug_info = True

REPO_NOTE = (
    "\n\nThe repository is checked out in /testbed, which is your working directory. "
    "Start by running `ls` to see the layout. Do not assume a src/ folder exists.\n"
)
# The default example made the model copy `find src ...` 29 times in our first run.
BAD_EXAMPLE = "find src -name '*.java' | grep -i builder"
GOOD_EXAMPLE = "find . -name '*.py' | grep -i builder"


def agent_config(cfg: Config) -> dict:
    """mini-swe-agent's SWE-bench config with our prompt fix and step limit."""
    config_dir = os.path.join(os.path.dirname(minisweagent.__file__), "config")
    config_file = glob.glob(f"{config_dir}/**/swebench.yaml", recursive=True)[0]
    agent_cfg = dict(yaml.safe_load(open(config_file)).get("agent", {}))
    agent_cfg["step_limit"] = cfg.step_limit
    for key, value in agent_cfg.items():
        if isinstance(value, str) and BAD_EXAMPLE in value:
            agent_cfg[key] = value.replace(BAD_EXAMPLE, GOOD_EXAMPLE)
    if "instance_template" not in agent_cfg:
        raise RuntimeError(f"mini-swe-agent config has no instance_template. Keys: {list(agent_cfg)}")
    agent_cfg["instance_template"] += REPO_NOTE
    return agent_cfg


def make_model(cfg: Config, seed: int) -> LitellmModel:
    sampling = cfg.active_sampling
    top_k = sampling.pop("top_k", None)
    extra_body = {**({"top_k": top_k} if top_k is not None else {}), **cfg.chat_template_kwargs()}
    return LitellmModel(
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
        return result


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
