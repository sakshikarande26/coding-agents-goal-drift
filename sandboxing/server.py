"""Start the vLLM model server and wait until it answers.

If a server already answers at `server.url`, it is reused. This also lets you run
against a server you started yourself (set `server.start: false`).
"""

from __future__ import annotations

import subprocess
import time
from urllib.parse import urlparse

import requests

from .config import Config


def server_is_up(url: str) -> bool:
    try:
        return requests.get(f"{url}/models", timeout=2).ok
    except requests.RequestException:
        return False


def start(cfg: Config) -> subprocess.Popen | None:
    """Return the server process we started, or None if one was already running."""
    url = cfg.server.url
    if server_is_up(url):
        print(f"Model server already up at {url}")
        return None
    if not cfg.server.start:
        raise RuntimeError(f"No model server at {url} and server.start is false")

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / "vllm.log"
    port = urlparse(url).port or 8000
    args = [
        "vllm", "serve", cfg.model, "--port", str(port),
        "--dtype", cfg.server.dtype,
        "--max-model-len", str(cfg.server.max_model_len),
        "--gpu-memory-utilization", str(cfg.server.gpu_memory_utilization),
        "--enforce-eager",
        # mini-swe-agent v2 uses tool calling
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]
    log = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)

    start_time = time.time()
    while time.time() - start_time < cfg.server.startup_timeout:
        if server_is_up(url):
            print(f"Model server up after {round(time.time() - start_time)} s")
            return proc
        if proc.poll() is not None:
            break
        time.sleep(10)
    stop(proc)
    raise RuntimeError(f"Model server did not start. Last log lines:\n{log_path.read_text()[-3000:]}")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def ping(cfg: Config) -> str:
    """Ask the model one short question to check the server and the thinking switch."""
    reply = requests.post(f"{cfg.server.url}/chat/completions", json={
        "model": cfg.model,
        "messages": [{"role": "user", "content": "Say hi in 3 words."}],
        "max_tokens": 512 if cfg.thinking else 20,
        **cfg.chat_template_kwargs(),
    }, timeout=120)
    reply.raise_for_status()
    return reply.json()["choices"][0]["message"]["content"]
