 LLM coding agents fix software issues over many steps, but they are trained with a single reward at the end: did the tests pass? That signal can't tell which
  step went wrong, so agents drift. They overreach (editing unrelated files, expanding scope) or fall into scope collapse (skipping requirements, gaming the
  tests), and the tests can still pass. This project builds step-level preference data (on-track vs. drifted next steps) and compares two ways to use it: a
  **reward model that reranks candidate actions** at each step, and **DPO fine-tuning** of the agent itself. We measure which one reduces drift more without
  lowering task success, using mini-SWE-agent with Qwen3 on R2E-Gym and SWE-bench Verified tasks.
