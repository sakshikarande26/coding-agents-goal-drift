import copy, json, random, time, requests
from dataclasses import dataclass, field, asdict

BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}   # replace with the harness's real tool schema

def commands_of(message):
    """All commands in an assistant message's tool calls, in order.
    None if there is no tool call, or if any call is malformed (treated as not runnable)."""
    cmds = []
    for tc in message.get("tool_calls") or []:
        try:
            args = tc["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            c = args["command"]
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(c, str) or not c.strip():
            return None
        cmds.append(c)
    return cmds or None

def _norm(command):
    return " ".join(command.split())

# ---- scorers: anything with score(prefix, action) -> float, called once per command ----
class RandomScorer:
    """Stand-in for an untrained reward model (also the 'unguided' selector)."""
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def score(self, prefix, action): return self.rng.random()

class RepeatPenaltyScorer:
    """Stub rule: penalize a command already run in this prefix. Demo only, not David's rules."""
    def score(self, prefix, action):
        seen = [_norm(c) for m in prefix for c in (commands_of(m) or [])]
        return -float(sum(_norm(action) == s for s in seen))

# ---- sampling and selection ----
def scrape_prefix_cache(url):
    """Best-effort vLLM prefix-cache counters from /metrics, summed over labels. {} if unavailable.
    Check the names on your server first: curl localhost:8000/metrics | grep prefix_cache"""
    try:
        text = requests.get(url.rsplit("/v1", 1)[0] + "/metrics", timeout=5).text
        out = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{")[0].split(" ")[0]
            if "prefix_cache" in name and name.endswith("_total"):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        return out
    except (requests.RequestException, ValueError):
        return {}

def _with_reasoning_content(m):
    """vLLM may name the parsed reasoning field 'reasoning'; the project schema uses 'reasoning_content'."""
    if m.get("reasoning") and not m.get("reasoning_content"):
        m = {**m, "reasoning_content": m["reasoning"]}
    return m

def sample_candidates(url, model, prefix, n, tools=(BASH_TOOL,), temperature=0.7, top_p=0.8,
                      top_k=20, presence_penalty=1.5, thinking=False, max_tokens=None,
                      seed=None, timeout=600):
    """Returns (assistant messages, info). info: finish_reasons, prompt_tokens, completion_tokens
    (summed over the n candidates; the prompt is counted once), seconds (request wall time)."""
    body = {"model": model, "messages": prefix, "tools": list(tools), "n": n,
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens or (4096 if thinking else 512),
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    r = requests.post(f"{url}/chat/completions", json=body, timeout=timeout)
    seconds = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    info = {"finish_reasons": [c.get("finish_reason") for c in data["choices"]],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "seconds": seconds}
    return [_with_reasoning_content(c["message"]) for c in data["choices"]], info

@dataclass
class StepLog:
    n_requested: int
    candidates: list          # per candidate: list of commands, None if no valid tool call
    scores: list              # per candidate: MIN score over its commands, None if invalid
    chosen_idx: int
    no_valid: bool = False
    n_distinct: int = 0       # distinct valid candidates (normalized command sequences)
    finish_reasons: list = field(default_factory=list)   # "length" means a truncated candidate
    prompt_tokens: int = 0
    completion_tokens: int = 0    # summed over all candidates
    sample_s: float = 0.0         # sampling request time: prefill + decode + queueing, set by the slowest candidate
    score_s: float = 0.0
    cache: dict = field(default_factory=dict)   # prefix-cache counter deltas (only if track_cache)

    @property
    def completion_tok_per_s(self):
        return self.completion_tokens / self.sample_s if self.sample_s else 0.0

def select(prefix, messages, scorer):
    cmds = [commands_of(m) for m in messages]
    cache, scores = {}, []
    for cs in cmds:
        if cs is None:
            scores.append(None)
            continue
        per = []
        for c in cs:                         # score every command in the turn, not just the first
            key = _norm(c)                   # identical commands are scored once
            if key not in cache:
                cache[key] = float(scorer.score(prefix, c))
            per.append(cache[key])
        scores.append(min(per))              # a turn is only as good as its worst command
    valid = [i for i, s in enumerate(scores) if s is not None]
    n_distinct = len({tuple(_norm(c) for c in cs) for cs in cmds if cs})
    if not valid:                            # nothing runnable: fall back to the first sample
        return messages[0], StepLog(len(messages), cmds, scores, 0, no_valid=True, n_distinct=n_distinct)
    best = max(valid, key=lambda i: (scores[i], -i))   # ties go to the lowest index
    return messages[best], StepLog(len(messages), cmds, scores, best, n_distinct=n_distinct)

def rerank_step(url, model, prefix, scorer, n=8, track_cache=False, **sampling):
    """One agent step. Returns (assistant message to append, StepLog, all raw candidate messages).
    track_cache: diff vLLM prefix-cache counters around the call (valid only if nothing else hits the server)."""
    before = scrape_prefix_cache(url) if track_cache else {}
    messages, info = sample_candidates(url, model, prefix, n, **sampling)
    t0 = time.perf_counter()
    chosen, log = select(prefix, messages, scorer)
    log.score_s = time.perf_counter() - t0
    after = scrape_prefix_cache(url) if track_cache else {}
    log.cache = {k: after[k] - before.get(k, 0.0) for k in after}
    log.finish_reasons = info["finish_reasons"]
    log.prompt_tokens, log.completion_tokens = info["prompt_tokens"], info["completion_tokens"]
    log.sample_s = info["seconds"]
    return chosen, log, messages             # keep all raw messages: they are the raw material for pairs

def write_log(path, log, **context):
    """Append one JSON line per step. context: traj_id, step, condition, thinking, gpu, ..."""
    rec = {**context, **asdict(log), "total_s": log.sample_s + log.score_s,
           "completion_tok_per_s": log.completion_tok_per_s}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ---- helpers for testing on a saved trajectory ----
KEEP = {"role", "content", "tool_calls", "tool_call_id", "name"}

def clean_messages(msgs):
    out = []
    for m in msgs:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        m = copy.deepcopy({k: v for k, v in m.items() if k in KEEP})   # never mutate the parent trajectory
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
        out.append(m)
    return out

def completed_turn_ends(clean):
    """Index just past each completed turn: an assistant message with tool calls plus all its tool results."""
    ends, i = [], 0
    while i < len(clean):
        if clean[i]["role"] == "assistant" and clean[i].get("tool_calls"):
            j = i + 1
            while j < len(clean) and clean[j]["role"] == "tool":
                j += 1
            ends.append(j)
            i = j
        else:
            i += 1
    return ends

def prefix_after_steps(clean, k):
    """Messages up to the end of the k-th completed turn (k=0: before the first assistant turn).
    Later user messages (e.g. format-error notices) are not included; the real pair generator
    should use data_eval.branching.shared_prefix with prefix_message_count instead."""
    if k == 0:
        return clean[:next(i for i, m in enumerate(clean) if m["role"] == "assistant")]
    return clean[:completed_turn_ends(clean)[k - 1]]

def _fake(*cmds):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({"command": c})}}
                           for c in cmds]}

def selftest():                              # no server needed
    prefix = [_fake("ls")]
    msgs = [_fake("ls"), {"role": "assistant", "content": "no tool"}, _fake("pwd"), _fake("ls")]
    chosen, log = select(prefix, msgs, RepeatPenaltyScorer())
    assert log.chosen_idx == 2 and log.scores == [-1.0, None, 0.0, -1.0], log
    _, log = select(prefix, [msgs[1]], RepeatPenaltyScorer())
    assert log.no_valid
    # multi-call: a repeat in the SECOND call must count (the old first-call-only code picked index 0)
    _, log = select(prefix, [_fake("pwd", "ls"), _fake("pwd")], RepeatPenaltyScorer())
    assert log.scores == [-1.0, 0.0] and log.chosen_idx == 1, log
    # malformed arguments anywhere in the turn make the candidate invalid
    bad = {"role": "assistant", "tool_calls": [{"function": {"arguments": "{oops"}}]}
    assert commands_of(bad) is None
    assert commands_of({"role": "assistant", "tool_calls": _fake("ls")["tool_calls"] + bad["tool_calls"]}) is None
    # diversity counts distinct normalized command sequences
    _, log = select([], [_fake("a  b"), _fake("a b"), _fake("c")], RandomScorer())
    assert log.n_distinct == 2
    # clean_messages must not mutate its input (pairs need the parent's raw messages byte for byte)
    raw = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]}]
    before = json.dumps(raw)
    clean_messages(raw)
    assert json.dumps(raw) == before
    # prefixes cut after ALL tool results of a multi-call turn
    cl = [{"role": "system"}, {"role": "user"}, _fake("a", "b"), {"role": "tool"}, {"role": "tool"},
          _fake("c"), {"role": "tool"}, {"role": "user"}]
    assert completed_turn_ends(cl) == [5, 7] and len(prefix_after_steps(cl, 1)) == 5 and len(prefix_after_steps(cl, 0)) == 2
    assert _with_reasoning_content({"role": "assistant", "reasoning": "x"})["reasoning_content"] == "x"
    print("selftest ok")

if __name__ == "__main__":
    import sys
    selftest()
    if len(sys.argv) > 1:                    # python rerank.py traj.json  (needs a vLLM server)
        URL, MODEL = "http://localhost:8000/v1", "Qwen/Qwen3-8B"
        # matches the cook_8b_v2 rollouts: thinking on, temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.5
        SAMPLING = dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)
        clean = clean_messages(json.load(open(sys.argv[1]))["messages"])
        n_turns = len(completed_turn_ends(clean))
        for k in (4, 10):
            if k > n_turns:
                print(f"step {k}: trajectory has only {n_turns} completed turns, skipping")
                continue
            prefix = prefix_after_steps(clean, k)
            _, log, msgs = rerank_step(URL, MODEL, prefix, RepeatPenaltyScorer(), n=8,
                                       track_cache=True, **SAMPLING)
            write_log("rerank_log.jsonl", log, step=k, thinking=True)
            print(f"step {k}: chosen {log.chosen_idx}, scores {log.scores}, finish {log.finish_reasons}")
            print(f"   {log.n_distinct} distinct, prompt {log.prompt_tokens} tok, {log.completion_tokens} completion tokens in "
                  f"{log.sample_s:.1f}s ({log.completion_tok_per_s:.0f} tok/s), cache {log.cache}")
            print("   message keys:", sorted(msgs[0].keys()), "| reasoning captured:",
                  any(m.get("reasoning_content") for m in msgs))
            print("   picked:", log.candidates[log.chosen_idx])