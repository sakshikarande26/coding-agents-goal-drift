"""Command line entry point.

    goal-drift smoke   --config sandboxing/pilot.yaml            # sandbox + grading only, no GPU
    goal-drift run     --config sandboxing/pilot.yaml            # full rollouts
    goal-drift doctor  --config sandboxing/pilot.yaml            # which backend, GPU and server this machine has
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import grading, trajectory
from .config import Config, load_config
from .sandbox import Sandbox
from .tasks import Task, load_tasks


def cmd_smoke(cfg: Config, parallel: int) -> int:
    """Per task: smoke test + sandbox checks. With --parallel K, the first task runs
    in K sandboxes at once to show per-run names do not collide."""
    tasks = load_tasks(cfg)
    all_ok = True
    for i, task in enumerate(tasks):
        print(f"\n=== {task.repo} {task.task_id[:8]} ===")
        print(Sandbox(cfg, task.image, "pull").pull())
        copies = parallel if i == 0 else 1

        def one(k: int) -> tuple[dict, dict]:
            run_id = f"{task.task_id[:12]}__smoke{k}__pid{os.getpid()}"
            return grading.smoke_test(Sandbox(cfg, task.image, run_id), task), \
                grading.sandbox_checks(Sandbox(cfg, task.image, run_id + "_checks"))

        with ThreadPoolExecutor(max_workers=copies) as pool:
            results = list(pool.map(one, range(copies)))
        for k, (smoke, checks) in enumerate(results):
            print(f"[copy {k}] BUGGY {smoke['buggy']} | FIXED {smoke['fixed']} | patch files "
                  f"{smoke['patch_files']} vs gold {smoke['gold_files']} -> {'OK' if smoke['ok'] else 'FAIL'}")
            for name, ok in checks.items():
                print(f"[copy {k}]   {'PASS' if ok else 'FAIL'} {name}")
            all_ok &= smoke["ok"] and all(checks.values())
        if cfg.cleanup_images:
            Sandbox(cfg, task.image, "pull").remove_image()
    print("\nALL OK" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


def run_one(cfg: Config, task: Task, seed: int, versions: dict) -> dict:
    """One task x one seed: fresh container, agent, grade, finalize, save."""
    from .agent import RepeatedFailure, make_agent

    traj_id = trajectory.make_traj_id(task.task_id, cfg, seed)
    sb = Sandbox(cfg, task.image, traj_id)
    try:
        sb.fresh()
        sb.hide_tests()
        agent, env = make_agent(cfg, sb, seed)
        start, exit_status, error = time.time(), None, None
        try:
            result = agent.run(task.problem_statement)
            exit_status = result.get("exit_status") if isinstance(result, dict) else str(result)
        except RepeatedFailure:
            exit_status = "RepeatedFailure"
        except Exception as e:
            exit_status, error = f"Error: {type(e).__name__}", str(e)[:1000]
        finally:
            sb.restore_tests()
        if env.stopped:
            exit_status = "RepeatedFailure"
        seconds = round(time.time() - start)

        patch = grading.get_patch(sb)
        output = grading.run_tests(sb)
        matched, mismatches = grading.grade(output, task.expected)
        traj = {
            "traj_id": traj_id,
            "parent_traj_id": None,  # set by branch_from()
            "branch_step": None,
            "task_id": task.task_id,
            "split": cfg.split,
            "repo": task.repo,
            "docker_image": task.image,
            "problem_statement": task.problem_statement,
            "condition": cfg.condition,
            "model": cfg.model,
            "thinking": cfg.thinking,
            "sampling": {**cfg.active_sampling, "seed": seed},
            "step_limit": cfg.step_limit,
            "max_repeats": cfg.max_repeats,
            "versions": versions,
            "finished_at": datetime.datetime.now().isoformat(),
            "seconds": seconds,
            "exit_status": exit_status,
            "error": error,
            "steps": 0,  # counted by finalize()
            "resolved": not mismatches,
            "tests_matched": f"{matched}/{len(task.expected)}",
            "test_summary": grading.summary(output),
            "ref_patch_files": task.gold_files(),
            "files_touched": grading.edited_files(patch),  # reverted edits are only in messages
            "patch": patch,
            "messages": agent.messages,  # untruncated: DPO needs the exact prompt and completions
        }
    finally:
        sb.cleanup()
    traj = trajectory.finalize(traj)
    path = trajectory.save(traj, cfg.trajs_dir)
    print(f"{exit_status} | {traj['steps']} steps | {seconds}s | resolved={traj['resolved']} "
          f"| touched {traj['files_touched']} -> {path.name}")
    return traj


def cmd_run(cfg: Config) -> int:
    from . import server
    from .agent import get_versions

    tasks = load_tasks(cfg)
    proc = server.start(cfg)
    try:
        print("Model says:", server.ping(cfg)[:200])
        versions = get_versions(cfg)
        print(versions)
        done, skipped = [], []
        for task in tasks:
            print(f"\n=== {task.repo} {task.task_id[:8]} ===")
            todo = [s for s in cfg.seeds
                    if not trajectory.out_path(cfg.trajs_dir, trajectory.make_traj_id(task.task_id, cfg, s)).exists()]
            if not todo:
                print("All seeds already saved, skipping")
                continue
            puller = Sandbox(cfg, task.image, "pull")
            puller.pull()
            smoke = grading.smoke_test(Sandbox(cfg, task.image, f"{task.task_id[:12]}__smoke__pid{os.getpid()}"), task)
            print(f"Smoke test: BUGGY {smoke['buggy']} | FIXED {smoke['fixed']} -> {'OK' if smoke['ok'] else 'FAIL'}")
            if not smoke["ok"]:
                skipped.append(task.task_id)
                continue
            for seed in todo:
                try:
                    done.append(run_one(cfg, task, seed, versions))
                except Exception as e:
                    print(f"Run crashed (seed {seed}): {type(e).__name__}: {str(e)[:300]}")
            if cfg.cleanup_images:
                puller.remove_image()
    finally:
        server.stop(proc)

    rows = trajectory.write_summary(cfg.trajs_dir) if cfg.trajs_dir.exists() else []
    if rows:
        resolved = sum(r["resolved"] for r in rows)
        print(f"\nPass rate: {resolved}/{len(rows)}")
        print(f"Runs with repeated failing commands: {sum(r['repeated_failures'] > 0 for r in rows)}/{len(rows)}")
        print(f"Runs editing outside the gold files: {sum(r['outside_gold'] > 0 for r in rows)}/{len(rows)}")
    print("Skipped (smoke test failed):", [t[:8] for t in skipped])
    return 0


def cmd_doctor(cfg: Config) -> int:
    """Print what this machine has, so you know which backend and settings will be used."""
    from .config import docker_works
    from .sandbox import sh
    from .server import server_is_up

    def found(tool: str) -> str:
        return shutil.which(tool) or "not found"

    gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=30) if shutil.which("nvidia-smi") else ""
    rows = [
        ("backend (used)", cfg.backend),
        ("apptainer", found(cfg.apptainer_bin)),
        ("singularity", found("singularity") + ("  (not used: cannot load Docker images on COOK)" if shutil.which("singularity") else "")),
        ("docker", found("docker") + ("  (daemon answers)" if docker_works() else "  (no working daemon)" if shutil.which("docker") else "")),
        ("udocker", found("udocker")),
        ("GPU", gpu.replace("\n", " | ") if gpu and "failed" not in gpu.lower() else "none (on a login node? start a job first)"),
        ("vllm installed", "yes" if importlib.util.find_spec("vllm") else "no"),
        ("model server", f"{cfg.server.url} {'up' if server_is_up(cfg.server.url) else 'not running'}"),
        ("model / thinking", f"{cfg.model} / {cfg.thinking}  sampling {cfg.active_sampling}"),
        ("results", str(cfg.trajs_dir)),
        ("work_dir", f"{cfg.work_dir}{'  (inside $HOME)' if cfg.work_dir.is_relative_to(Path.home()) else ''}"),
        ("validator", "data_eval.validate_trajectories imported OK"),
    ]
    for name, value in rows:
        print(f"{name:18} {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goal-drift", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run", "doctor"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.yaml"))
        p.add_argument("--n-tasks", type=int, help="override n_tasks")
        p.add_argument("--work-dir", type=str, help="override work_dir")
        p.add_argument("--run-name", type=str, help="override run_name (results/<run_name>/)")
        p.add_argument("--steps", type=int, help="override step_limit")
        p.add_argument("--no-thinking", action="store_true", help="run with thinking off")
        p.add_argument("--server-url", type=str,
                       help="use a model server that is already running, e.g. llama-server on COOK")
        if name == "smoke":
            p.add_argument("--parallel", type=int, default=1, help="run the first task in K sandboxes at once")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, n_tasks=args.n_tasks, work_dir=args.work_dir, run_name=args.run_name,
                      step_limit=args.steps, thinking=False if args.no_thinking else None,
                      server_url=args.server_url)

    if args.command == "smoke":
        return cmd_smoke(cfg, args.parallel)
    if args.command == "doctor":
        return cmd_doctor(cfg)
    return cmd_run(cfg)


if __name__ == "__main__":
    sys.exit(main())
