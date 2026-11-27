#!/usr/bin/env python3
"""Generate candidate action sets (N per prefix) and two-candidate A/B items for labeling.

  python generate_candidates.py cook_8b_v2_pilot15/ --repo-root ~/proj/repo --out out_pilot \
      --per-traj 2 --n 8 --url http://localhost:8000/v1 --model Qwen/Qwen3-8B

Needs a running vLLM server (tool parser + reasoning parser, see run_generate_candidates.sbatch) and the
team repo, because prefixes are built with data_eval.branching.shared_prefix(messages, prefix_len),
assumed to return (prefix_messages, error) like validate_pairs.py uses it.

Outputs in --out (JSONL, appended, so a preempted job can be rerun and skips finished prefixes):
  candidate_sets.jsonl  all N raw candidates per prefix (keep these: David wants the full sets saved)
  pair_items.jsonl      two-candidate A/B items; no drift labels or chosen/rejected yet
  step_log.jsonl        tokens, wall time, finish reasons, prefix-cache deltas per prefix
  skipped.jsonl         prefixes that produced no pair, with the reason
"""
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

import rerank

SKIP_NAMES = {"manifest.json", "pair_manifest.json", "drift_flags.json"}


class ConstantScorer:
    """Candidate generation does not rank; rerank_step just needs some scorer."""
    def score(self, prefix, action): return 0.0


def load_shared_prefix(repo_root):
    sys.path.insert(0, str(Path(repo_root).expanduser()))
    try:
        from data_eval.branching import shared_prefix
    except ImportError:
        sys.path.insert(0, str(Path(repo_root).expanduser() / "data_eval"))
        try:
            from branching import shared_prefix
        except ImportError:
            sys.exit(f"Could not import shared_prefix from {repo_root}/data_eval/branching.py; pass --repo-root.")
    return shared_prefix


def load_trajectories(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix in (".json", ".jsonl") and f.name not in SKIP_NAMES)
        else:
            files.append(p)
    for f in files:
        text = f.read_text(encoding="utf-8")
        recs = [json.loads(l) for l in text.splitlines() if l.strip()] if f.suffix == ".jsonl" else [json.loads(text)]
        for r in recs:
            if isinstance(r, dict) and r.get("traj_id") and isinstance(r.get("messages"), list):
                yield r


def prefix_tokens(messages, k):
    """Prompt tokens the original rollout saw for turn k+1 (saved usage), else a rough chars/3.5 estimate."""
    asst = [m for m in messages if m.get("role") == "assistant"]
    try:
        return asst[k]["extra"]["response"]["usage"]["prompt_tokens"]
    except (IndexError, KeyError, TypeError):
        end = rerank.completed_turn_ends(rerank.clean_messages(messages))[k - 1]
        return int(sum(len(json.dumps(m)) for m in messages[:end]) / 3.5)


def pick_prefixes(messages, per_traj, min_k, max_tokens):
    """Evenly spaced prefix_len values (completed turns), leaving at least one later step, within the token limit."""
    n_turns = len(rerank.completed_turn_ends(rerank.clean_messages(messages)))
    ks = [k for k in range(min_k, n_turns) if prefix_tokens(messages, k) <= max_tokens]
    if not ks or per_traj >= len(ks):
        return ks
    if per_traj == 1:
        return [ks[len(ks) // 2]]
    return sorted({ks[round(i * (len(ks) - 1) / (per_traj - 1))] for i in range(per_traj)})


def slim(msg):
    """Keep only what the pair schema and DPO loader need; vLLM adds null fields and a duplicate 'reasoning'."""
    out = {"role": "assistant", "content": msg.get("content") or ""}
    rc = msg.get("reasoning_content") or msg.get("reasoning")
    if rc:
        out["reasoning_content"] = rc
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        call = {"type": "function", "function": {"name": fn.get("name"),
                "arguments": args if isinstance(args, str) else json.dumps(args)}}
        if tc.get("id"):
            call["id"] = tc["id"]
        calls.append(call)
    if calls:
        out["tool_calls"] = calls
    return out


def make_pairs(log, n_pairs, rng):
    """Pick pairs of valid candidates with different command sequences; A/B order is randomized."""
    first_of = {}
    for i, cs in enumerate(log.candidates):
        fr = log.finish_reasons[i] if i < len(log.finish_reasons) else None
        if cs is None or fr == "length":
            continue
        first_of.setdefault(tuple(rerank._norm(c) for c in cs), i)   # one representative per distinct sequence
    reps = sorted(first_of.values())
    combos = [(a, b) for j, a in enumerate(reps) for b in reps[j + 1:]]
    rng.shuffle(combos)
    pairs = []
    for a, b in combos[:n_pairs]:
        pairs.append((b, a) if rng.random() < 0.5 else (a, b))
    return pairs


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(args, sampler=rerank.rerank_step):
    shared_prefix = None if args.dry_run else load_shared_prefix(args.repo_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets_path, pairs_path = out / "candidate_sets.jsonl", out / "pair_items.jsonl"
    done = set()
    if sets_path.exists():
        done = {json.loads(l)["candidate_set_id"] for l in sets_path.read_text().splitlines() if l.strip()}
    sampling = dict(thinking=not args.no_thinking, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, presence_penalty=args.presence_penalty, max_tokens=args.max_tokens)
    provenance = {"generator": "generate_candidates.py v0", "engine": "vllm", "model": args.model,
                  "tool_schema_sha": hashlib.sha256(json.dumps(rerank.BASH_TOOL, sort_keys=True).encode()).hexdigest()[:12],
                  "n_requested": args.n, "sampling_seed": None, "gpu": args.gpu, **{
                      k: sampling[k] for k in ("thinking", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens")}}
    stats = Counter()
    bp = None
    if args.branch_points:
        bp = {}
        for p in json.load(open(args.branch_points))["points"]:
            bp.setdefault(p["traj_id"], []).append(p["branch_step"])
    for traj in load_trajectories(args.inputs):
        tid, messages = traj["traj_id"], traj["messages"]
        if traj.get("split") not in args.splits:
            stats["skipped_split"] += 1
            continue
        if bp is not None:
            if tid not in bp:
                stats["skipped_not_in_branch_points"] += 1
                continue
            ks = sorted(bp[tid])
        else:
            ks = pick_prefixes(messages, args.per_traj, args.min_k, args.max_prompt_tokens)
        for k in ks:
            set_id = f"{tid}:k{k}"
            if set_id in done:
                stats["already_done"] += 1
                continue
            if args.dry_run:
                print(f"{set_id}  repo={traj.get('repo')}  ~{prefix_tokens(messages, k)} prompt tokens")
                stats["planned"] += 1
                continue
            prefix, err = shared_prefix(messages, k)
            if err or not prefix:
                append_jsonl(out / "skipped.jsonl", {"candidate_set_id": set_id, "reason": f"shared_prefix: {err}"})
                stats["skipped_prefix"] += 1
                continue
            _, log, msgs = sampler(args.url, args.model, rerank.clean_messages(prefix), ConstantScorer(),
                                   n=args.n, track_cache=not args.no_track_cache, **sampling)
            rerank.write_log(out / "step_log.jsonl", log, candidate_set_id=set_id, parent_traj_id=tid,
                             prefix_len=k, gpu=args.gpu)
            cands = [slim(m) for m in msgs]
            append_jsonl(sets_path, {
                "candidate_set_id": set_id, "parent_traj_id": tid, "task_id": traj.get("task_id"),
                "split": traj.get("split"), "prefix_len": k, "candidates": cands, "commands": log.candidates,
                "finish_reasons": log.finish_reasons, "n_distinct": log.n_distinct, "provenance": provenance,
                "usage": {"prompt_tokens": log.prompt_tokens, "completion_tokens": log.completion_tokens,
                          "sample_s": round(log.sample_s, 2)}})
            stats["candidate_sets"] += 1
            pairs = make_pairs(log, args.pairs_per_set, random.Random(f"{args.seed}:{set_id}"))
            if not pairs:
                append_jsonl(out / "skipped.jsonl", {"candidate_set_id": set_id,
                             "reason": f"fewer than 2 distinct valid candidates (n_distinct={log.n_distinct})"})
                stats["no_pair"] += 1
            for a, b in pairs:
                append_jsonl(pairs_path, {
                    "pair_id": f"{set_id}:{a}-{b}", "candidate_set_id": set_id, "parent_traj_id": tid,
                    "task_id": traj.get("task_id"), "split": traj.get("split"), "prefix_len": k,
                    "prompt_messages": prefix, "candidate_a": cands[a], "candidate_b": cands[b],
                    "a_idx": a, "b_idx": b, "provenance": provenance})
                stats["pair_items"] += 1
    print(dict(stats))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="trajectory JSON/JSONL files or directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]),
                    help="repo containing data_eval/ (default: parent of this script's folder)")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--per-traj", type=int, default=2, help="prefixes per trajectory")
    ap.add_argument("--branch-points", default=None, help="branch_points_pilot.json: use exactly these (traj_id, branch_step) prefixes")
    ap.add_argument("--pairs-per-set", type=int, default=1)
    ap.add_argument("--min-k", type=int, default=2, help="smallest prefix_len to use")
    ap.add_argument("--max-prompt-tokens", type=int, default=14000)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--seed", type=int, default=0, help="seeds A/B order and pair choice only, not vLLM sampling")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--max-tokens", type=int, default=None, help="default 4096 with thinking, 512 without")
    ap.add_argument("--gpu", default=None, help="recorded in provenance")
    ap.add_argument("--no-track-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the planned prefixes, no server needed")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
