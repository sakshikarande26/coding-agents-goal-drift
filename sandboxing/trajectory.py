"""Trajectory records: ids, finalize() against David's validator, saving,
and the drift-indicator summary."""

from __future__ import annotations

import json
from pathlib import Path

# David's validator is the contract. finalize() calls it; never copy its field list.
from data_eval.validate_trajectories import validate_trajectory

from .config import Config
from .sandbox import safe_name


class InvalidTrajectory(ValueError):
    def __init__(self, traj_id, errors):
        self.errors = errors
        super().__init__(f"{traj_id}: " + "; ".join(errors))


def make_traj_id(task_id: str, cfg: Config, seed: int) -> str:
    """Same inputs give the same id, so a crashed batch can be re-run and skips finished runs."""
    model_short = cfg.model.split("/")[-1]
    return f"{task_id[:12]}__{cfg.condition}__{model_short}__think{int(cfg.thinking)}__seed{seed}"


def finalize(traj: dict) -> dict:
    """Fill defaults, fix known name differences, then validate with David's validator.

    Returns exactly what will be saved. Raises InvalidTrajectory if anything is wrong.
    """
    t = json.loads(json.dumps(traj, default=str))  # validate the JSON form that gets saved
    t.setdefault("parent_traj_id", None)
    t.setdefault("branch_step", None)
    t.setdefault("error", None)

    versions = {str(k): str(v) for k, v in (t.get("versions") or {}).items()}
    if "chat_template_sha256" in versions:  # v3 notebook name
        versions.setdefault("chat_template_hash", versions.pop("chat_template_sha256"))
    for key in ("model_revision", "chat_template_hash"):
        if key not in versions and t.get(key):
            versions[key] = str(t[key])
        if key in versions:
            t[key] = versions[key]  # also top level, as in CONTEXT.md section 8
    t["versions"] = versions

    for key in ("files_touched", "ref_patch_files"):
        if isinstance(t.get(key), list):
            t[key] = sorted(set(t[key]))
    if isinstance(t.get("messages"), list):
        t["steps"] = sum(1 for m in t["messages"] if isinstance(m, dict) and m.get("role") == "assistant")

    errors, warnings = validate_trajectory(t)
    for warning in warnings:
        print(f"  warning ({t.get('traj_id')}): {warning}")
    if errors:
        raise InvalidTrajectory(t.get("traj_id"), errors)
    return t


def out_path(out_dir: Path, traj_id: str) -> Path:
    return out_dir / f"{safe_name(traj_id)}.json"


def save(traj: dict, out_dir: Path) -> Path:
    """Write atomically, so a crash never leaves a half-written file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_path(out_dir, traj["traj_id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(traj, indent=1))
    tmp.replace(path)
    return path


# --- summary table ---

def _commands_and_codes(messages: list) -> list[dict]:
    steps, pending = [], []
    for m in messages:
        extra = m.get("extra") or {}
        if m.get("role") == "assistant":
            pending += [a.get("command") for a in extra.get("actions", [])]
        elif m.get("role") == "tool":
            steps.append({"command": pending.pop(0) if pending else None, "returncode": extra.get("returncode")})
    return steps


def drift_indicators(traj: dict) -> dict:
    """Quick automatic indicators for the summary table (not labels)."""
    steps = _commands_and_codes(traj["messages"])
    failed = [s["returncode"] not in (0, None) for s in steps]
    repeats = sum(
        1 for i in range(1, len(steps))
        if failed[i] and failed[i - 1] and steps[i]["command"] == steps[i - 1]["command"]
    )
    gold, edited = set(traj["ref_patch_files"]), set(traj["files_touched"])
    return {
        "traj_id": traj["traj_id"],
        "repo": traj["repo"],
        "thinking": traj["thinking"],
        "seed": traj["sampling"]["seed"],
        "exit": traj["exit_status"],
        "steps": len(steps),
        "failed_cmds": sum(failed),
        "repeated_failures": repeats,
        "edited": len(edited),
        "outside_gold": len(edited - gold),
        "touched_gold": bool(edited & gold),
        "resolved": traj["resolved"],
        "seconds": traj.get("seconds"),
    }


def summarize(trajs_dir: Path) -> list[dict]:
    """Drift indicators for every trajectory JSON in the folder (printed, not saved)."""
    return [drift_indicators(json.loads(p.read_text())) for p in sorted(trajs_dir.glob("*.json"))]
