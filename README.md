# Mitigating Goal Drift in Coding Agents

LLM coding agents fix software issues over many steps, but they are trained with a single reward at the end: did the tests pass? That signal can't tell which step went wrong, so agents drift. They overreach (editing unrelated files, expanding scope) or fall into scope collapse (skipping requirements, gaming the tests), and the tests can still pass. This project builds step-level preference data (on-track vs. drifted next steps) and compares two ways to use it: a **reward model that reranks candidate actions** at each step, and **DPO fine-tuning** of the agent itself. We measure which one reduces drift more without lowering task success, using mini-SWE-agent with Qwen3 on R2E-Gym and SWE-bench Verified tasks.

Project context, team split and decisions: [CONTEXT.md](CONTEXT.md).

## Layout

```
sandboxing/             Track 1: rollout harness (Sakshi, Khoa)
  pilot.yaml            all run settings
  config.py             load + check the settings
  tasks.py              R2E-Gym loading, answer key, gold fix
  sandbox.py            apptainer / docker / udocker containers, one per run
  grading.py            run tests, grade, patch, smoke test, sandbox checks
  server.py             start vLLM and wait until it answers
  agent.py              prompt fix, model client, environment + repeat guard
  trajectory.py         finalize() with David's validator, save, summary
  run.py                command line: doctor, smoke, run
data_eval/              Track 2: drift rules, validators, branching helpers (David)
```

Run outputs go to `~/goal-drift-work/` (set by `work_dir`), outside the repo: `trajs/`, `runs/`, `sifs/`, `logs/`.

## Setup

```bash
git clone https://github.com/sakshikarande26/coding-agents-goal-drift.git
cd coding-agents-goal-drift
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,udocker]"     # Colab
pip install -e ".[vllm]"             # Unity (apptainer comes from `module load apptainer`)
```

`backend: auto` picks apptainer on Unity, docker on a machine where it runs, and udocker on Colab and COOK.
On Colab, run `udocker --allow-root install` once (on COOK: `udocker install`).
On Unity, set `backend: apptainer` and a `/work/...` path for `work_dir` in `sandboxing/pilot.yaml`, and pre-pull SIFs into `<work_dir>/sifs/` in a CPU job.

## Commands

Run from the repo root.

```bash
goal-drift doctor --config sandboxing/pilot.yaml                 # what this machine has: backend, GPU, server
goal-drift smoke --config sandboxing/pilot.yaml                 # no GPU: sandbox + grading checks
goal-drift smoke --config sandboxing/pilot.yaml --parallel 2    # also checks two runs at once don't collide
goal-drift run   --config sandboxing/pilot.yaml                 # full rollouts (needs a GPU for vLLM)
python data_eval/validate_trajectories.py ~/goal-drift-work/trajs         # David's validator + manifest
```

`run` skips (task, seed) pairs that already have a file in `<work_dir>/trajs/`, so a crashed job can be re-run.

## What a run produces

`<work_dir>/trajs/<traj_id>.json`, one per task x seed, plus `<work_dir>/trajs/summary.csv`.
`traj_id` = `<task>__<condition>__<model>__think<0|1>__seed<N>`.
Every file goes through `finalize()`, which runs David's `validate_trajectory()` and refuses to save an invalid record.

## Rules the harness keeps

- The gold patch and the answer key are never shown to the agent. They are used only in the smoke test and in grading.
- Hidden tests are moved out of the container while the agent runs.
- The patch is diffed against a snapshot taken right before the agent starts, not against `HEAD`.
- Each run has its own container (`r2e_<traj_id>`) and folder (`<work_dir>/runs/<traj_id>/`), so runs can go in parallel.
