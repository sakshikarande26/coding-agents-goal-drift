#!/usr/bin/env python3
"""Turn human annotations into chosen/rejected training pairs.

  python labels_to_pairs.py --items out_pilot_X/pair_items.jsonl --annotations annotations.csv --out pairs_labeled.jsonl

Annotations CSV (David's v2 format, one row per annotator per item, one column per drift type per candidate):
  pair_id,annotator_id,preference,
  a_loop,a_unrelated_edit,a_scope_creep,a_edits_tests,a_hardcode,a_skipped_requirement,
  b_loop,b_unrelated_edit,b_scope_creep,b_edits_tests,b_hardcode,b_skipped_requirement,notes
  preference: A, B, tie or uncertain.
  drift cells: 1/yes/true (type present), 0/no/false/none (absent), uncertain (cannot judge this type).
  A blank or unrecognised drift cell is treated like 'uncertain' and counted in a warning at the end.
pair_id must match the pair_id in the items file.

Consensus rules (change with the flags if the team decides otherwise):
  * preference: the most common of A / B / tie among raters; 'uncertain' abstains. Excluded if there is no strict
    majority, the majority is tie, or the winner has fewer than --min-votes votes.
  * drift, decided separately for each type on each candidate: raters who said 'uncertain' for THAT type abstain.
    Among the remaining raters, the type is 'present' on a strict majority, 'absent' on a strict majority of no,
    and 'unresolved' otherwise (an even split, or fewer than --min-drift-raters decisive raters).
    Unresolved types are not added to the drift list and are reported in *_drift_unresolved.
  * an item is excluded if it has fewer than --n-raters annotators (use --n-raters 0 to disable this check).
Output pairs carry chosen_drift / rejected_drift lists (confirmed present types only; empty = none confirmed),
chosen_drift_unresolved / rejected_drift_unresolved lists, label_source 'human', and the vote counts.
Items without a consensus are written to <out>.excluded.jsonl with the reason.
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict

SIX = ["loop", "unrelated_edit", "scope_creep", "edits_tests", "hardcode", "skipped_requirement"]
YES = {"1", "y", "yes", "true", "x"}
NO = {"0", "n", "no", "false", "none"}
UNC = {"uncertain", "u", "?", "unc"}
BAD_CELLS = []          # (pair_id, annotator, column, raw value) for blank or unrecognised drift cells


def parse_cell(cell, where):
    """'yes', 'no' or 'unc'. Blank or unrecognised values count as 'unc' and are recorded as warnings."""
    c = (cell or "").strip().lower()
    if c in YES:
        return "yes"
    if c in NO:
        return "no"
    if c in UNC:
        return "unc"
    BAD_CELLS.append((*where, cell))
    return "unc"


def consensus_pref(votes, min_votes):
    c = Counter(v for v in votes if v in ("a", "b", "tie"))
    if not c:
        return None, "no decisive votes"
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, "no majority"
    winner, n = top[0]
    if winner == "tie":
        return None, "majority tie"
    if n < min_votes:
        return None, f"only {n} vote(s) for the winner"
    return winner, None


def candidate_drift(rows, side, min_raters):
    """(present types, unresolved types) for one candidate, one type at a time."""
    present, unresolved = [], []
    for t in SIX:
        col = f"{side}_{t}"
        vals = [parse_cell(r.get(col), (r["pair_id"], r["annotator_id"], col)) for r in rows]
        valid = [v for v in vals if v != "unc"]
        yes = sum(v == "yes" for v in valid)
        no = len(valid) - yes
        if len(valid) < min_raters:
            unresolved.append(t)
        elif yes * 2 > len(valid):
            present.append(t)
        elif no * 2 > len(valid):
            pass                      # confirmed absent
        else:
            unresolved.append(t)      # even split
    return present, unresolved


def load_annotations(paths):
    by_pair = defaultdict(dict)       # pair_id -> annotator -> row
    need = ["pair_id", "annotator_id", "preference"] + [f"{s}_{t}" for s in "ab" for t in SIX]
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in need if c not in (reader.fieldnames or [])]
            if missing:
                sys.exit(f"{p}: missing column(s): {', '.join(missing)}")
            for row in reader:
                pid, ann = row["pair_id"].strip(), row["annotator_id"].strip()
                if ann in by_pair[pid]:
                    print(f"warning: duplicate row for {pid} / {ann}; the later row wins", file=sys.stderr)
                row["pair_id"], row["annotator_id"] = pid, ann
                by_pair[pid][ann] = row
    return by_pair


def convert(items, ann, min_votes=2, n_raters=3, min_drift_raters=2):
    pairs, excluded = [], []
    for pid, rows_by_ann in ann.items():
        rows = list(rows_by_ann.values())
        item = items.get(pid)
        if item is None:
            excluded.append({"pair_id": pid, "reason": "pair_id not found in items"})
            continue
        if n_raters and len(rows) < n_raters:
            excluded.append({"pair_id": pid, "reason": f"only {len(rows)} of {n_raters} raters"})
            continue
        votes = [r["preference"].strip().lower() for r in rows]
        winner, why = consensus_pref(votes, min_votes)
        if winner is None:
            excluded.append({"pair_id": pid, "reason": why, "votes": dict(Counter(votes))})
            continue
        a_drift, a_unres = candidate_drift(rows, "a", min_drift_raters)
        b_drift, b_unres = candidate_drift(rows, "b", min_drift_raters)
        a_first = winner == "a"
        chosen, rejected = (item["candidate_a"], item["candidate_b"]) if a_first else (item["candidate_b"], item["candidate_a"])
        c_drift, r_drift = (a_drift, b_drift) if a_first else (b_drift, a_drift)
        c_unres, r_unres = (a_unres, b_unres) if a_first else (b_unres, a_unres)
        rec = {k: item[k] for k in ("pair_id", "task_id", "parent_traj_id", "prefix_len", "prompt_messages", "split",
                                     "candidate_set_id") if k in item}
        rec.update(chosen=chosen, rejected=rejected,
                   chosen_idx=item["a_idx"] if a_first else item["b_idx"],
                   rejected_idx=item["b_idx"] if a_first else item["a_idx"],
                   label_source="human",
                   chosen_drift=c_drift, rejected_drift=r_drift,
                   chosen_drift_unresolved=c_unres, rejected_drift_unresolved=r_unres,
                   drift_uncertain=bool(c_unres or r_unres),
                   votes=dict(Counter(votes)), n_raters=len(rows),
                   notes=[f"{r['annotator_id']}: {r['notes'].strip()}" for r in rows if (r.get("notes") or "").strip()],
                   provenance=item.get("provenance"))
        pairs.append(rec)
    return pairs, excluded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", required=True, help="pair_items.jsonl file(s)")
    ap.add_argument("--annotations", nargs="+", required=True, help="annotation CSV file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-votes", type=int, default=2, help="votes the preference winner needs")
    ap.add_argument("--n-raters", type=int, default=3, help="annotators required per item (0 disables the check)")
    ap.add_argument("--min-drift-raters", type=int, default=2,
                    help="decisive (non-uncertain) raters needed to resolve one drift type on one candidate")
    args = ap.parse_args()
    items = {}
    for p in args.items:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                it = json.loads(l)
                items[it["pair_id"]] = it
    ann = load_annotations(args.annotations)
    pairs, excluded = convert(items, ann, args.min_votes, args.n_raters, args.min_drift_raters)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".excluded.jsonl", "w", encoding="utf-8") as f:
        for r in excluded:
            f.write(json.dumps(r) + "\n")
    unannotated = sorted(set(items) - set(ann))
    print(f"{len(items)} items, {len(ann)} annotated, {len(unannotated)} with no annotations")
    for pid in unannotated[:30]:
        print("  unannotated", pid)
    print(f"{len(pairs)} pairs written to {args.out}; {len(excluded)} excluded")
    for r in excluded:
        print("  excluded", r["pair_id"], "-", r["reason"])
    if pairs:
        unres = Counter(t for p in pairs for t in p["chosen_drift_unresolved"] + p["rejected_drift_unresolved"])
        print("unresolved drift cells by type (across kept pairs):", dict(unres) or "none")
        for p in pairs:
            if p["chosen_drift_unresolved"] or p["rejected_drift_unresolved"]:
                print(f"  review {p['pair_id']}: chosen {p['chosen_drift_unresolved'] or '-'}, "
                      f"rejected {p['rejected_drift_unresolved'] or '-'}")
    if BAD_CELLS:
        print(f"warning: {len(BAD_CELLS)} blank or unrecognised drift cell(s) treated as uncertain, e.g. {BAD_CELLS[:3]}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
