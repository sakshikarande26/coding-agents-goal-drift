 LLM coding agents fix software issues over many steps, but they are trained with a single reward at the end: did the tests pass? That signal can't tell which
  step went wrong, so agents drift. They overreach (editing unrelated files, expanding scope) or fall into scope collapse (skipping requirements, gaming the
  tests), and the tests can still pass. This project builds step-level preference data (on-track vs. drifted next steps) and compares two ways to use it: a
  **reward model that reranks candidate actions** at each step, and **DPO fine-tuning** of the agent itself. We measure which one reduces drift more without
  lowering task success, using mini-SWE-agent with Qwen3 on R2E-Gym and SWE-bench Verified tasks.

## How it works

1. Load 5 small R2E-Gym bug-fix tasks (1 file, ≤10 changed lines, ≤2 per repo).
2. Open a sealed sandbox per run with the buggy code.
3. Fairness check: buggy code fails the tests, the known fix passes (e.g. aiohttp: 13/14 → 14/14).
4. Hide the tests. The agent (Qwen3-4B, thinking on) runs one shell command per step, up to 30 steps; it is stopped after 5 identical failing commands.
5. Restore the tests, run them, record pass or fail.
6. Save one JSON per run to `results/<run_name>/`. `data_eval/` checks it and flags drift.

## Files

| `sandboxing/` (agent runs) | |
|---|---|
| `pilot.yaml` | All settings |
| `config.py` | Reads and checks the settings |
| `tasks.py` | Loads tasks, answer key and correct fix (never shown to the agent) |
| `sandbox.py` | One sealed container per run (apptainer, docker or udocker) |
| `grading.py` | Runs tests, scores, shows the agent's changes, 4 sandbox checks |
| `server.py` | Starts the model server |
| `agent.py` | The agent loop, thinking switch, repeat guard |
| `trajectory.py` | Builds, validates (David's checker) and saves the JSON |
| `run.py` | Commands: `doctor`, `smoke`, `run` |

| `data_eval/` (checks the JSON) | |
|---|---|
| `validate_trajectories.py` | Checks every field of a run file |
| `branching.py` | Counts steps, extracts the shared start of a run |
| `drift_rules.py` | 3 of 6 drift rules: loop (3rd identical failed command), test edits, unrelated edits |
| `offline_analysis.py` | Flags drift in finished runs |
| `live_filter.py` | Rule-filter baseline: picks 1 of N commands (not connected yet) |
| `validate_pairs.py` | Checks good-step / bad-step pairs |

`results/` holds run outputs, `CONTEXT.md` the full plan. Scratch files (images, hidden tests) go to `~/goal-drift-work/`, outside the repo.

## Run the agent (`sandboxing/`)

```bash
pip install -e ".[udocker]"     # add vllm on a machine that serves the model: pip install -e ".[udocker,vllm]"
udocker install                 # once (on Colab: udocker --allow-root install). Not needed on Unity (apptainer)

goal-drift doctor               # shows backend, GPU, model server and results folder for this machine
goal-drift sample               # makes the frozen task list (already done: sandboxing/task_list.json)
goal-drift smoke                # checks the sandbox and grading; no GPU needed
goal-drift run                  # starts vLLM, runs the agent, saves results/<run_name>/*.json
```

Settings are in `sandboxing/pilot.yaml` (small test) and `sandboxing/cook_8b.yaml` (the real batch). Common flags: `--n-tasks`, `--split train|heldout`, `--seeds 0 1`, `--shard i/n`, `--steps`, `--no-thinking`, `--run-name`, `--server-url`.

**Task list.** `sandboxing/task_list.json` is frozen: small fixes from R2E-Gym-Lite (at most 1 non-test file and 10 lines, 1,597 tasks), then 15 train + 9 heldout per repo, picked at random with `fixed_seed: 2026` (any fixed number gives the same tasks every time), so 150 train + 90 heldout. Tasks are in repo round-robin order, so the first 100 train tasks are 10 per repo. Do not regenerate it; heldout tasks are for the final eval only.

- **Unity:** set `work_dir` in `pilot.yaml` to a `/work/...` path.
- **COOK** (vLLM does not run on its GPUs): `sandboxing/cook_run.sbatch` runs one llama-server and one shard of the tasks per GPU (see the comments at its top).

## Results

[`results/cook_8b_v2/`](results/cook_8b_v2/) is the trajectory set for making pairs: 165 runs. That is seed 0 on 88 of the first 100 train tasks (8 to 10 per repo), plus seeds 1 to 4 on 29 of them from an earlier, larger plan. The other 12 tasks failed the smoke test: 7 because our test-output parser missed colored output and test names with spaces (fixed in `grading.py`, to be rerun), 2 need internet, 2 have a wrong answer key in R2E-Gym, 1 has a flaky test. Qwen3-8B (Q8_0, llama-server on COOK), thinking on, `unguided`, 30-step limit, repeat guard 5. Every file passes `data_eval/validate_trajectories.py`. Runs that end with `Submitted` get a warning: the final submit command has no tool output, because mini-swe-agent stops there. Branch only from these train runs.

[`results/cook_dev/`](results/cook_dev/) has one trajectory on the aiohttp task (thinking on, 30-step limit). The agent was stopped after 8 steps by the repeat guard, in 81 s, without fixing the bug (13/14 tests). It passes `data_eval/validate_trajectories.py`.

Made on COOK with a compressed Qwen3-4B (4-bit, llama-server). Use it to build and test code (validator, drift rules, branching, pairs), not as report numbers.
