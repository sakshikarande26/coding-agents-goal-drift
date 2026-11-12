 LLM coding agents fix software issues over many steps, but they are trained with a single reward at the end: did the tests pass? That signal can't tell which
  step went wrong, so agents drift. They overreach (editing unrelated files, expanding scope) or fall into scope collapse (skipping requirements, gaming the
  tests), and the tests can still pass. This project builds step-level preference data (on-track vs. drifted next steps) and compares two ways to use it: a
  **reward model that reranks candidate actions** at each step, and **DPO fine-tuning** of the agent itself. We measure which one reduces drift more without
  lowering task success, using mini-SWE-agent with Qwen3 on R2E-Gym and SWE-bench Verified tasks.

## Project structure

| Path | What it is |
|---|---|
| `sandboxing/` | Runs the coding agent on R2E-Gym bug-fix tasks inside a container and saves a record of every step (one JSON file per run) |
| `sandboxing/pilot.yaml` | All run settings: backend, model, thinking on/off, tasks, seeds, step limit |
| `sandboxing/run.py` | Command line: `goal-drift doctor`, `goal-drift smoke`, `goal-drift run` |
| `data_eval/` | Checks those run records: flags drift (loops, test edits, unrelated edits) and validates the file format |
| `CONTEXT.md` | Full project context: plan, team split, decisions, findings |
| `results/<run_name>/` | Run outputs: one trajectory JSON per run plus `summary.csv` (folder name set by `run_name` in `pilot.yaml`) |
| `pyproject.toml` | Dependencies; install with `pip install -e .` |

Scratch files (container images, hidden tests) go to `~/goal-drift-work/`, outside the repo.

## Run the agent (`sandboxing/`)

```bash
pip install -e ".[udocker]"     # add vllm on a machine that serves the model: pip install -e ".[udocker,vllm]"
udocker install                 # once (on Colab: udocker --allow-root install). Not needed on Unity (apptainer)

goal-drift doctor               # shows backend, GPU, model server and results folder for this machine
goal-drift smoke                # checks the sandbox and grading; no GPU needed
goal-drift run                  # starts vLLM, runs the agent, saves results/<run_name>/*.json
```

Settings are in `sandboxing/pilot.yaml`. Common flags: `--n-tasks`, `--steps`, `--no-thinking`, `--run-name`.

- **Unity:** set `work_dir` in `pilot.yaml` to a `/work/...` path.
- **COOK** (vLLM does not run on its GPUs): start `llama-server` on port 8080, then `goal-drift run --server-url http://localhost:8080/v1 --run-name cook_dev`.

## Results

[`results/cook_test/`](results/cook_test/) has two trajectories on the aiohttp task, one with thinking on and one with thinking off, plus `summary.csv`. Each file passes `data_eval/validate_trajectories.py`.

They were made on COOK with a compressed Qwen3-4B (4-bit, llama-server) and only 6–8 steps. Use them to build and test code (validator, drift rules, branching, pairs), not as report numbers.
