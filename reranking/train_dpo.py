#!/usr/bin/env python3
"""LoRA DPO on step-level preference pairs (Qwen3-8B, TRL + PEFT).

  python train_dpo.py --pairs pairs_labeled.jsonl --out dpo_run1
  python train_dpo.py --smoke-items out_pilot_X/pair_items.jsonl --out dpo_smoke --max-steps 6   # plumbing test only

Pair records: prompt_messages (exact parent history), chosen and rejected (assistant message dicts with
content, reasoning_content, tool_calls), task_id or parent_traj_id, optional split.
--smoke-items turns unlabeled A/B items into pairs with a RANDOM winner. That only tests that the pipeline runs and
gives real step times and memory; the resulting adapter is meaningless.

Prompts are rendered with the model's chat template plus the same bash tool schema used for candidate generation
(rerank.BASH_TOOL), so training prompts match inference prompts. History messages drop reasoning_content (as in
candidate generation) unless --keep-history-reasoning. The chosen/rejected completions keep their reasoning.
"""
import argparse, json, random, sys
from pathlib import Path

from rerank import BASH_TOOL


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm_message(m, keep_reasoning):
    """Template-ready message. Tool-call arguments must be dicts for HF chat templates, not JSON strings."""
    content = m.get("content") or ""
    if isinstance(content, list):
        content = " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    out = {"role": m["role"], "content": content}
    if m["role"] == "tool" and m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if keep_reasoning and m.get("reasoning_content"):
        out["reasoning_content"] = m["reasoning_content"]
    calls = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args or {}}})
    if calls:
        out["tool_calls"] = calls
    return out


def build_example(tok, pair, tools=(BASH_TOOL,), keep_history_reasoning=False):
    """Returns ({prompt, chosen, rejected} as plain text, None) or (None, reason).
    Completion = rendered assistant message minus the final EOS, which TRL appends itself."""
    prompt = [norm_message(m, keep_history_reasoning) for m in pair["prompt_messages"]]
    p_text = tok.apply_chat_template(prompt, tools=list(tools), add_generation_prompt=True, tokenize=False)
    comps = []
    for x in (pair["chosen"], pair["rejected"]):
        msg = x if isinstance(x, dict) else {"role": "assistant", "content": x}
        full = tok.apply_chat_template(prompt + [norm_message(msg, True)], tools=list(tools), tokenize=False)
        if not full.startswith(p_text):
            return None, "rendered conversation does not extend the rendered prompt"
        c = full[len(p_text):].rstrip("\n")
        if tok.eos_token and c.endswith(tok.eos_token):
            c = c[: -len(tok.eos_token)]
        comps.append(c)
    if comps[0] == comps[1]:
        return None, "chosen and rejected render identically"
    return {"prompt": p_text, "chosen": comps[0], "rejected": comps[1]}, None


def smoke_pairs(items, seed):
    rng = random.Random(seed)
    out = []
    for it in items:
        a, b = it["candidate_a"], it["candidate_b"]
        if rng.random() < 0.5:
            a, b = b, a
        out.append({"pair_id": it["pair_id"], "task_id": it.get("task_id"), "parent_traj_id": it.get("parent_traj_id"),
                    "split": it.get("split", "train"), "prompt_messages": it["prompt_messages"],
                    "chosen": a, "rejected": b, "label_source": "random_smoke"})
    return out


def group_split(rows, groups, eval_frac, seed):
    """Split by task/trajectory so prefixes of one run never sit in both train and eval."""
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_eval = max(1, round(len(uniq) * eval_frac)) if eval_frac > 0 and len(uniq) > 1 else 0
    ev = set(uniq[:n_eval])
    tr_rows = [r for r, g in zip(rows, groups) if g not in ev]
    ev_rows = [r for r, g in zip(rows, groups) if g in ev]
    return tr_rows, ev_rows


def filter_kwargs(cls, kw):
    import inspect
    valid = set(inspect.signature(cls).parameters)
    if "eval_strategy" in kw and "eval_strategy" not in valid and "evaluation_strategy" in valid:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = sorted(k for k in kw if k not in valid)
    if dropped:
        print(f"note: {cls.__name__} does not accept {dropped}; ignored")
    return {k: v for k, v in kw.items() if k in valid}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs", help="labeled pairs JSONL (chosen/rejected schema)")
    src.add_argument("--smoke-items", help="pair_items.jsonl; random winners, pipeline test only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--split", default="train", help="only pairs from this split are used")
    ap.add_argument("--keep-history-reasoning", action="store_true")
    ap.add_argument("--max-total-tokens", type=int, default=14000, help="drop pairs longer than this")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    pairs = smoke_pairs(load_jsonl(args.smoke_items), args.seed) if args.smoke_items else load_jsonl(args.pairs)
    if args.smoke_items:
        print("SMOKE MODE: random winners. The adapter is meaningless; this only tests the pipeline.")
    pairs = [p for p in pairs if p.get("split", "train") == args.split]
    tok = AutoTokenizer.from_pretrained(args.model)
    rows, groups, dropped = [], [], {}
    for p in pairs:
        ex, why = build_example(tok, p, keep_history_reasoning=args.keep_history_reasoning)
        if ex is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        n = max(len(tok(ex["prompt"] + ex[k], add_special_tokens=False).input_ids) for k in ("chosen", "rejected"))
        if n > args.max_total_tokens:
            dropped["too long"] = dropped.get("too long", 0) + 1
            continue
        rows.append(ex)
        groups.append(p.get("task_id") or p.get("parent_traj_id") or p.get("pair_id"))
    print(f"{len(rows)} usable pairs from {len(pairs)}; dropped: {dropped}")
    if len(rows) < 4:
        sys.exit("Too few usable pairs.")
    tr, ev = group_split(rows, groups, args.eval_frac, args.seed)
    print(f"train {len(tr)}, eval {len(ev)} (split by task/trajectory)")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    cfg = DPOConfig(**filter_kwargs(DPOConfig, dict(
        output_dir=args.out, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=args.max_steps, beta=args.beta, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=1, eval_strategy="epoch" if ev else "no", save_strategy="no", report_to="none",
        max_length=args.max_total_tokens + 512, max_prompt_length=args.max_total_tokens, seed=args.seed)))
    trainer_kw = dict(model=model, args=cfg, train_dataset=Dataset.from_list(tr),
                      eval_dataset=Dataset.from_list(ev) if ev else None, peft_config=lora)
    try:
        trainer = DPOTrainer(processing_class=tok, **trainer_kw)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tok, **trainer_kw)
    trainer.train()
    final_eval = trainer.evaluate() if ev else {}
    trainer.save_model(str(Path(args.out) / "adapter"))

    log = trainer.state.log_history
    losses = [e["loss"] for e in log if "loss" in e]
    summary = {"n_train": len(tr), "n_eval": len(ev), "first_train_loss": losses[0] if losses else None,
               "last_train_loss": losses[-1] if losses else None, "final_eval": final_eval,
               "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "train_runtime_s": next((e.get("train_runtime") for e in reversed(log) if "train_runtime" in e), None),
               "gpu": torch.cuda.get_device_name(0), "args": vars(args), "smoke": bool(args.smoke_items)}
    Path(args.out, "log_history.json").write_text(json.dumps(log, indent=1))
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
