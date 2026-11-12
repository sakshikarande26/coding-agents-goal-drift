# CS 690S Project Context: Mitigating Goal Drift in Coding Agents

**Team:** Khoa Ho, Sakshi Karande, David Makarovsky
**Course:** COMPSCI 690S AI Alignment, UMass Amherst
**Last updated:** 2026-11-12, evening (after the notebook moved to `sandboxing/`)

---

## 1. Project in one paragraph

LLM coding agents that fix repository issues over many steps, with only a pass/fail signal at the end, can drift: they edit unrelated files, expand scope, or repeat a failing fix. Outcome-only RL (e.g. FrogNano) gives one binary reward per trajectory and cannot say which step went wrong. We build **step-level preference pairs** (a good next action vs a drifting next action from the same point in a trajectory) and compare two ways of using them on the same data: **(a)** a Bradley–Terry **reward model that reranks** N candidate actions at each step, with the agent frozen, and **(b)** **LoRA DPO** that fine-tunes the agent itself. To our knowledge, prior work has not compared these head to head on identical pairs and baselines.

## 2. Research question and hypotheses

**Question:** Given the same step-level preference data, which mitigates goal drift better: an external step-level reward model that reranks actions, or direct policy fine-tuning with DPO?

- **H1:** Reward-model reranking reduces human-judged drift compared with (i) matched-compute unguided sampling and (ii) the rule-based filter, without a pass-rate drop whose confidence interval excludes zero.
- **H2:** On the longer-horizon subset (above-median trajectory length), reranking has lower drift than DPO. Mechanism: DPO shifts the policy mainly on prefixes resembling the pair data, while a reward model scores any recognizable state. The opposite is plausible and will be reported.

**Success criteria:** H1 holds if the drift-difference interval excludes zero against both baselines and pass rate does not drop significantly. H2 holds if reranking has lower drift on the long subset with an interval excluding zero.

**Stretch:** sweep N from 1 to 16 against human labels to test reward-model overoptimization.

## 3. Key terms

| Term | Meaning |
|---|---|
| Trajectory | One full agent run: every message, command, output, the final patch and the test result |
| Step | One assistant turn (a thought plus one bash command) and the tool output it gets back |
| Prefix | Everything in a trajectory up to the step being judged |
| Branching | Restarting from a saved prefix and sampling different next actions |
| Preference pair | (prefix, chosen action, rejected action), where both actions share the same prefix |
| `branch_step` | Number of completed assistant+tool pairs in the shared prefix (David's definition) |
| Gold / reference patch | The real commit that fixed the bug, from the dataset. Used for the smoke test and drift labels, never shown to the agent |
| Patch | The agent's code changes as a `git diff`, measured against the starting snapshot |
| Drift | Actions that move away from the task: looping, editing tests, unrelated edits, scope creep, hardcoding, skipped requirements |
| Matched compute | Baselines get the same number of samples or tokens as the reranker |
| SIF | Apptainer's single-file, read-only container image |

## 4. System design and flow

```
Task prep ──► Rollout engine ──► Pair builder ──► Training ──► Selector arms ──► Evaluation
R2E-Gym       mini-SWE-agent     branch, label,   RM (BT)      random, rule,     pass rate, drift,
filter 20-80% + vLLM + sandbox   hand-check       DPO (LoRA)   RM, DPO           bootstrap
```

1. **Task prep.** Load R2E-Gym, run pilot rollouts, keep tasks solved 20–80% of the time. Fixed split: ~150 train, ~60 held-out, 30 SWE-bench Verified.
2. **Rollout engine.** vLLM serves the policy. mini-SWE-agent runs one bash command per step inside a per-task container. Tests are hidden during the run and restored for grading.
3. **Pair builder.** Replay a saved prefix in a fresh container, sample K alternative next actions, auto-label drift, and validate a subset by three-way hand labeling (Fleiss' kappa).
4. **Training.** Bradley–Terry reward model (small model + scalar head) and LoRA DPO on the policy, both on the **same pair file**.
5. **Selector arms.** One harness with a swappable `Selector.pick(prefix, candidates)`:
   - unguided (N=1)
   - random of N (matched compute)
   - rule-based filter (frozen before eval)
   - untrained RM
   - trained RM rerank
   - DPO policy (N=1)
6. **Evaluation.** Test pass rate, blind human-judged drift rate, inference cost, RM accuracy on held-out pairs. Paired bootstrap over tasks, 3 seeds, long-horizon split for H2.

**Design principle:** every arm runs through the same harness and differs in one component (the selector or the weights). That is the experimental control.

## 5. Track 1: Infra and rollouts

**Owners:** Sakshi (harness, `sandboxing/`), Khoa (Unity port and testing)

### Built
- **Sandbox, Colab:** udocker runs R2E-Gym Docker images without Docker. Smoke test on aiohttp task `fca11627`: buggy 13/14, gold fix 14/14.
- **Sandbox, Unity:** Apptainer. 410 MB SIF, about 10 min first pull, 13/14 as expected. A per-run writable sandbox works: edits persist across exec calls, the baseline-snapshot diff is clean, hide/restore of `/r2e_tests` works.
- **Harness:** mini-SWE-agent 2.4.6 + vLLM 0.33 + Qwen3-4B on a T4. Tool calling via the hermes parser.
- **Notebook versions:**
  - **v1** (`sandboxing.ipynb`): one task end to end.
  - **v2:** patch baseline fix, prompt fix, `Qwen3-4B-Instruct-2507`, repeat guard, gold-fix test-path fix, 5-task loop, summary table with drift indicators.
  - **v3** (`sandboxing_v3.ipynb`): `BACKEND` switch (udocker / apptainer), Khoa's flags, sandbox checks (section 6b), new schema fields.
  - **Now: `sandboxing/` package** (replaces the notebooks, which are removed). Settings in `sandboxing/pilot.yaml`. Commands: `goal-drift doctor` (what this machine has), `goal-drift smoke` (sandbox + grading checks, no GPU), `goal-drift run` (rollouts). Flags: `--n-tasks`, `--steps`, `--no-thinking`, `--run-name`, `--server-url`.
- **Backends:** `backend: auto` picks apptainer (Unity), else docker if it runs, else udocker (Colab, COOK). Docker path is untested.
- **Outputs:** one trajectory JSON per run in `results/<run_name>/` (in the repo). Scratch (images, hidden tests, per-run copies) in `work_dir`, outside the repo.
- **Sample trajectory:** `results/cook_dev/` (aiohttp task, thinking on, 30-step limit; stopped after 8 steps by the repeat guard, not resolved). Made with a 4-bit Qwen3-4B on COOK: for building code, not report numbers.

### Bugs found and fixed
| Bug | Fix |
|---|---|
| Patch included R2E's own setup edits (image working tree differs from `HEAD`) | Snapshot the tree with `git write-tree` after container creation and diff against it |
| Agent copied `find src` from the default prompt example and looped 30 times | Example replaced; prompt says the repo is in `/testbed` and to start with `ls` |
| `presence_penalty` did not stop loops | Repeat guard: stop after 5 identical failing commands (`RepeatedFailure`) |
| Gold fix skipped any path containing "test" | Skip only files inside test folders |

### Khoa's Unity wrapper requirements (all in `sandboxing/sandbox.py`)
- `--pwd /testbed` (no chdir warnings in observations)
- `--silent`
- `--containall` (no host `$HOME`, `/tmp` or env vars such as `HF_TOKEN`)
- `GIT_PAGER=cat`
- Hidden tests, gold patches and trajectories kept under `WORK`, outside `$HOME` (with apptainer, the config refuses a `work_dir` inside `$HOME`)

### Sakshi, v4 status
1. **`finalize(traj)`:** done. Runs David's `validate_trajectory()` before every save and refuses invalid records. Allowed conditions are imported from `data_eval`. Both sample trajectories pass.
2. **Save the tools list:** open.
3. **Save all raw candidates per step** (`candidates_per_step`): open.
4. **Per-run container names and directories** (`r2e_<traj_id>`, one `Sandbox` object per run): done. Two runs at once tested on COOK.
5. **`THINKING` switch:** done (`thinking:` in `pilot.yaml` or `--no-thinking`; sampling preset per mode). With llama-server the thinking text is saved in each message's `reasoning_content`; with vLLM it may be inside `content`.
6. **`branch_from(traj, branch_step, k)`:** open, next. Must pass David's prefix check (`data_eval/branching.py`). Reuse Khoa's diversity script.
7. **Pilot** (4B vs 8B, thinking on vs off): open. Needs vLLM on Unity or Colab.

### Open (Khoa)
- Pre-pull all SIFs in a CPU-only batch job, time a second image pull, clean `$B/apptainer_cache`, check the /work quota.
- Test per-run overlays in place of 1.2 GB sandbox copies.
- Run `goal-drift smoke --parallel 2` on Unity with `work_dir` on `/work` (the apptainer path of the new code is untested).
- vLLM serve test with Qwen3-8B and the DPO length ladder (queued on Unity).

## 6. Track 2: Data and evaluation

**Owner:** David

### Built
- `drift_rules`, `live_filter`, `offline_analysis` in the repo.
- Rules implemented: **looping**, **editing tests**, **unrelated edits**.
- Trajectory schema validator. On the v1 trajectory it failed on 10 missing fields, all of which v3/v4 add.

### Open
- Rules still to add: **scope creep**, **hardcoding**, **skipped requirements**.
- `unrelated_edit` only fires when the issue names a source file. Measure how often that happens across the 150 tasks.
- Validate that a branch's prefix equals its parent's.
- Pair-file validator.
- Labeling guide, ~300 pairs, three-way hand labels, Fleiss' kappa.
- Paired bootstrap analysis and the N sweep.

### Schema decision
The **validator is the contract**. The generating script conforms to it through `finalize()`. Put the field list and types in one shared `schema.py` that both import, so they cannot drift apart.

## 7. Track 3: Training and serving

**Owner:** proposed by the teammate who wrote the track plan (confirm)

### Built
- vLLM serving on a Colab T4 with Qwen3-4B (fp16, 16k context, tool calling).

### Open
- **Reward model:** Qwen3-0.6B or 1.7B with LoRA and a scalar head. Bradley–Terry loss `-log σ(r_chosen − r_rejected)`. Split pairs by task (never by pair) to avoid leakage. Report held-out pairwise accuracy against the untrained RM (~50%).
- **Reranking loop:** a `Selector` subclass that samples N candidates, scores each (prefix, action), and executes the argmax.
- **DPO:** TRL `DPOTrainer` with LoRA on the same pair file. **Proof of concept by default** (see decisions).
- **Hardware notes:** T4 has no bf16 (use fp16). Unity `gpu-preempt` has L4 24 GB (8B in bf16 fits, modest throughput). Measure tokens/sec at N=4 before fixing the matched-compute budget.

## 8. Trajectory schema (agreed fields)

```
traj_id, parent_traj_id, branch_step, task_id, split, repo, docker_image,
problem_statement, condition, model, thinking,
sampling{temperature, top_p, top_k, presence_penalty, seed},
step_limit, max_repeats, versions{...}, model_revision, chat_template_hash,
finished_at, seconds, exit_status, error, steps, resolved, tests_matched, test_summary,
ref_patch_files, files_touched, patch, tools, candidates_per_step, messages[]
```

- `messages[]` is saved **untruncated**, including thinking and the exact observations (DPO needs the exact prompt and completions).
- `files_touched` comes from the final patch. Files edited and then reverted are only visible in `messages[]`.
- `split` is one of `train`, `heldout`, `verified`.
- `condition` is an enum. David's validator (the contract) allows `unguided`, `untrained_rm`, `rule_filter`, `rm_rerank`, `dpo`. **Open:** add `random_n` and `branch` (`branch_from()` needs `branch`).

## 9. Findings so far

1. **The sandbox and grading are correct** on both backends (13/14 → 14/14).
2. **Natural drift example:** the first Qwen3-4B run (thinking off) ran `find src ... | grep router` 29 times. 30 steps, 30 failed commands, 27 exact repeats, not resolved. It is a clean "repeated failing command" case.
3. **Diversity test (Khoa, one trajectory, N=8):**
   - Thinking off: 4–7 distinct commands at steps 0–1, then 1 distinct and 8/8 repeats from step 6. Temperature 1.0 does not help.
   - Thinking on: 2–8 distinct commands and no repeats at any step.
   - **Implication:** reranking needs candidate diversity, so thinking off gives the reward model nothing to choose between once the agent loops.
4. **COOK cluster (CICS, 1080 Ti GPUs):** udocker runs R2E images without root (smoke test passes). Docker is not available, and singularity fails on Docker images (`/etc/hosts` permission denied). vLLM 0.33 does not run on the 1080 Ti (`cudaErrorNoKernelImageForDevice`), so COOK is for development only, with llama-server. Jobs are limited to 2 hours.

## 10. Decisions and changes from the proposal

| Topic | Proposal | Now |
|---|---|---|
| GPU | Single A100 | Colab T4 for prototyping, Unity L4 for runs. A100 access pending |
| Policy model | Qwen3-8B (or 4B) | 4B on Colab, 8B on Unity. 10-task 4B vs 8B pilot pending |
| Thinking | Not specified | Proposed **on** for main runs, pending a 10-task pilot |
| DPO | Full arm, fallback to proof of concept if behind at 11/12 | **Proof of concept by default**, since pairs and the RM were not ready by 11/12 |
| Sandbox | Docker | udocker (Colab), Apptainer (Unity) |
| Drift categories | Out-of-gold edits, repeated failing commands | Six rule types (three implemented) |

## 11. Risks

- **Small effects:** FrogNano's ranking verifier gained about one point over random.
- **Label noise:** hence three-way hand validation.
- **Diversity collapse:** with thinking off, candidates become identical once the agent loops.
- **Compute:** Unity GPUs are scarce at peak, and pulling 150 images at ~10 min each must happen in CPU jobs.
- **Prefix replay:** flaky tests or nondeterministic installs can break branching.
- **Off-policy data:** if prefixes come from a different model than the one trained.

## 12. Timeline

| When | Milestone |
|---|---|
| 11/12 | Mid-project check-in |
| Done 11/12 | `sandboxing/` package: finalize, per-run names, thinking switch, backends |
| Next | v4 rest: tools list, candidates per step |
| Next | `branch_from()`, then the 10-task pilot (4B vs 8B, thinking on vs off) |
| Then | Pre-pull SIFs, pilot filter on ~150 train tasks |
| Then | ~300 pairs, hand labels, kappa |
| Then | Reward model, reranking arm, baselines |
| Then | DPO proof of concept |
| December | Evaluation (pass rate, drift, bootstrap, long-horizon split), final report |

## 13. Asks and blockers

- **Unity access for Sakshi and David:** Prof. Niekum's group (`pi_sniekum_umass_edu`) is in `superpod-a100`'s AllowAccounts. Ask him to add them to the group directly. HPC help: hpc@umass.edu.
- **Khoa's branching script:** share it so `branch_from()` reuses it.
- **Khoa:** test the new code on Unity (`goal-drift smoke --parallel 2`).
- **David:** add `random_n` and `branch` to `CONDITIONS`?

## 14. Resources

### Papers (from the proposal)
1. R. Shah et al. *Goal misgeneralization.* arXiv:2210.01790, 2022. https://arxiv.org/abs/2210.01790
2. M. Kim et al. *FrogNano.* arXiv:2609.07925, 2026. https://arxiv.org/abs/2609.07925
3. S. Gandhi et al. *Steer, don't solve.* arXiv:2606.21811, 2026. https://arxiv.org/abs/2606.21811
4. S. Gandhi et al. *When agents go astray.* arXiv:2509.02360, 2025. https://arxiv.org/abs/2509.02360
5. N. Jain et al. *R2E-Gym.* arXiv:2504.07164, 2025. https://arxiv.org/abs/2504.07164
6. C. Jimenez et al. *SWE-bench.* ICLR 2024.
7. H. Lightman et al. *Let's verify step by step.* ICLR 2024.
8. R. Rafailov et al. *Direct preference optimization.* NeurIPS 2023.
9. L. Gao et al. *Scaling laws for reward model overoptimization.* ICML 2023.

### Datasets
- `R2E-Gym/R2E-Gym-Lite` (used by the notebook), plus `R2E-Gym/R2E-Gym-Subset` and `R2E-Gym/R2E-Gym-Full`. Key fields: `problem_statement`, `docker_image`, `parsed_commit_content` (gold fix), `expected_output_json` (answer key), `modified_files`, `num_non_test_files`, `num_non_test_lines`.
- `princeton-nlp/SWE-bench_Verified` (evaluation set, `patch` field is the gold patch).
- `tarsur385/swebench-verified-trajectories`: existing mini-SWE-agent runs on SWE-bench Verified (10 models, 500 tasks, `resolved` flag). Useful for offline drift analysis.

### Code and tools
- mini-SWE-agent: https://github.com/SWE-agent/mini-swe-agent
- R2E-Gym: https://github.com/R2E-Gym/R2E-Gym
- vLLM (serving), TRL (`RewardTrainer`, `DPOTrainer`), PEFT (LoRA)
- udocker (Docker images without Docker, Colab and COOK), Apptainer (Unity)
- Models: `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B`

### Team files
- `sandboxing/`: the harness (replaces the v1–v3 notebooks, still in git history)
- `results/cook_dev/`: one sample trajectory in the current schema
- Repo: `drift_rules`, `live_filter`, `offline_analysis`, schema validator (David)
- Khoa's diversity test output (36-row CSV: step × temperature × thinking → distinct, no_action, same_as_prev)

### Compute
- Colab free T4 (16 GB, fp16 only, no Docker)
- Unity `gpu-preempt` L4 24 GB. `superpod-a100` via Prof. Niekum's group if access is granted
- COOK (CICS) `gtx` partition, GTX 1080 Ti 11 GB, 2-hour jobs. No vLLM; development only (see Findings)
