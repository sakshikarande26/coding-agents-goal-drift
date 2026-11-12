 LLM coding agents fix software issues over many steps, but they are trained with a single reward at the end: did the tests pass? That signal can't tell which
  step went wrong, so agents drift. They overreach (editing unrelated files, expanding scope) or fall into scope collapse (skipping requirements, gaming the
  tests), and the tests can still pass. This project builds step-level preference data (on-track vs. drifted next steps) and compares two ways to use it: a
  **reward model that reranks candidate actions** at each step, and **DPO fine-tuning** of the agent itself. We measure which one reduces drift more without
  lowering task success, using mini-SWE-agent with Qwen3 on R2E-Gym and SWE-bench Verified tasks.

## Project structure

| Path | What it is |
|---|---|
| `sandboxing/` | Track 1: rollout harness. Runs the agent on R2E-Gym tasks in a container and saves one trajectory JSON per run |
| `sandboxing/pilot.yaml` | All run settings: backend, model, thinking on/off, tasks, seeds, step limit |
| `sandboxing/run.py` | Command line: `goal-drift doctor`, `goal-drift smoke`, `goal-drift run` |
| `data_eval/` | Track 2: drift rules, rule-filter baseline, trajectory and pair validators |
| `CONTEXT.md` | Full project context: plan, team split, decisions, findings |
| `pyproject.toml` | Dependencies; install with `pip install -e .` |

Run outputs (trajectories, container images, hidden tests) are written to `~/goal-drift-work/`, outside the repo.
