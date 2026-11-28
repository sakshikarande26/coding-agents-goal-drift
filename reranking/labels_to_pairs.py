#!/usr/bin/env python3
"""Turn human annotations into chosen/rejected training pairs.

  python labels_to_pairs.py --items out_pilot_X/pair_items.jsonl --annotations r1.csv r2.csv r3.csv --out pairs_labeled.jsonl

Annotations CSV (David's v2 format, one row per annotator per item, one column per drift type per candidate):
  pair_id,annotator_id,preference,
  a_loop,a_unrelated_edit,a_scope_creep,a_edits_tests,a_hardcode,a_skipped_requirement,
  b_loop,b_unrelated_edit,b_scope_creep,b_edits_tests,b_hardcode,b_skipped_requirement,notes
  preference: A, B, tie or uncertain.
  drift cells: yes, no or uncertain (y/true/1 and n/false/0 are accepted as equivalents).
pair_id must match the pair_id in the items file.

Input checks match fleiss_kappa.py, so a CSV that passes one passes the other:
  * blank, partially filled or unrecognised cells are errors (file:line is reported), never silently 'no';
    --blank-as-uncertain relaxes this for PARTLY filled rows only (blank cells become 'uncertain', with a warning);
  * a completely blank row is an error; --allow-incomplete skips such placeholder rows;
  * a duplicate (pair_id, annotator_id) is an error;
  * the same annotators must label every complete pair.

Consensus rules (change with the flags if the team decides otherwise):
  * preference: the most common of A / B / tie among raters; 'uncertain' abstains. Excluded if there is no strict
    majority, the majority is tie, or the winner has fewer than --min-votes votes.
    (fleiss_kappa.py instead drops a pair from the PREFERENCE kappa if any rater said uncertain; that statistic
    and this training-data consensus answer different questions.)
  * drift, decided separately for each type on each candidate: raters who said 'uncertain' for THAT type abstain.
    Among the remaining raters, the type is 'present' on a strict majority of yes, 'absent' on a strict majority of
    no, and 'unresolved' otherwise (an even split, or fewer than --min-drift-raters decisive raters).
    Unresolved types are not added to the drift list and are reported in *_drift_unresolved.
  * an item is excluded if it has fewer than --n-raters annotators (use --n-raters 0 to disable this check).
Output pairs carry chosen_drift / rejected_drift lists (confirmed present types only; empty = none confirmed),
chosen_drift_unresolved / rejected_drift_unresolved lists, label_source 'human', and the vote counts.
Items without a consensus are written to <out>.excluded.jsonl with the reason.
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict

SIX = ["loop", "unrelated_edit", "scope_creep", "edits_tests", "hardcode", "skipped_requirement"]
YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}
PREFS = ("a", "b", "tie", "uncertain")
LABEL_COLS = ["preference"] + [f"{s}_{t}" for s in "ab" for t in SIX]
LENIENT_BLANKS = []     # locations of blank cells turned into 'uncertain' by --blank-as-uncertain


class AnnotationError(ValueError):
    pass


def parse_preference(value, loc, blank_ok):
    v = (value or "").strip().lower()
    if v in PREFS:
        return v
    if not v and blank_ok:
        LENIENT_BLANKS.append(loc)
        return "uncertain"
    raise AnnotationError(f"{loc}: preference {value!r} must be A, B, tie or uncertain")


def parse_drift(value, loc, blank_ok):
    """'yes', 'no' or 'unc'."""
    v = (value or "").strip().lower()
    if v in YES:
        return "yes"
    if v in NO:
        return "no"
    if v == "uncertain":
        return "unc"
    if not v and blank_ok:
        LENIENT_BLANKS.append(loc)
        return "unc"
    raise AnnotationError(f"{loc}: drift value {value!r} must be yes, no or uncertain (1/0 and true/false accepted)")


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
        valid = [r[side][t] for r in rows if r[side][t] != "unc"]
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


def load_annotations(paths, blank_as_uncertain=False, allow_incomplete=False):
    by_pair = defaultdict(dict)       # pair_id -> annotator -> parsed row
    skipped_blank = []
    need = ["pair_id", "annotator_id"] + LABEL_COLS
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = [(n or "").strip() for n in (reader.fieldnames or [])]
            if not fields:
                raise AnnotationError(f"{p}: CSV has no header")
            if len(fields) != len(set(fields)):
                raise AnnotationError(f"{p}: duplicate header column names")
            missing = [c for c in need if c not in fields]
            if missing:
                raise AnnotationError(f"{p}: missing column(s): {', '.join(missing)}")
            for line_no, raw in enumerate(reader, start=2):
                loc = f"{p}:{line_no}"
                if None in raw:
                    raise AnnotationError(f"{loc}: too many comma-separated fields")
                row = {(k or "").strip(): v for k, v in raw.items()}
                pid, ann = (row["pair_id"] or "").strip(), (row["annotator_id"] or "").strip()
                if not pid or not ann:
                    raise AnnotationError(f"{loc}: pair_id and annotator_id must be nonempty")
                if ann in by_pair[pid]:
                    raise AnnotationError(f"{loc}: duplicate annotation for pair {pid!r}, annotator {ann!r}")
                empties = [c for c in LABEL_COLS if not (row[c] or "").strip()]
                if len(empties) == len(LABEL_COLS):
                    if allow_incomplete:
                        skipped_blank.append(f"{pid}:{ann}")
                        by_pair[pid][ann] = None          # occupies the key so a later row is flagged as duplicate
                        continue
                    raise AnnotationError(f"{loc}: completely unlabeled row for {pid!r}; fill it in "
                                          "or use --allow-incomplete to skip placeholder rows")
                if empties and not blank_as_uncertain:
                    raise AnnotationError(f"{loc}: partially labeled row, empty: {', '.join(empties)}; "
                                          "enter yes/no/uncertain explicitly (or pass --blank-as-uncertain)")
                by_pair[pid][ann] = {
                    "preference": parse_preference(row["preference"], f"{loc} preference", blank_as_uncertain),
                    **{s: {t: parse_drift(row[f"{s}_{t}"], f"{loc} {s}_{t}", blank_as_uncertain) for t in SIX}
                       for s in "ab"},
                    "notes": (row.get("notes") or "").strip()}
    for pid in by_pair:                                    # drop the skipped placeholders
        by_pair[pid] = {a: r for a, r in by_pair[pid].items() if r is not None}
    return by_pair, skipped_blank


def check_annotator_sets(ann, n_raters):
    """Every complete pair must have the same annotators (fleiss_kappa.py requires this too)."""
    if not n_raters:
        return
    sets = {frozenset(rows) for rows in ann.values() if len(rows) == n_raters}
    if len(sets) > 1:
        raise AnnotationError("annotator sets differ between pairs: " + "; ".join(sorted(",".join(sorted(s)) for s in sets)))


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
        votes = [r["preference"] for r in rows]
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
                   notes=[f"{a}: {r['notes']}" for a, r in rows_by_ann.items() if r["notes"]],
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
    ap.add_argument("--blank-as-uncertain", action="store_true",
                    help="treat blank cells in PARTLY filled rows as 'uncertain' (default: error, as in fleiss_kappa.py)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="skip completely blank placeholder rows (default: error, as in fleiss_kappa.py)")
    args = ap.parse_args()
    items = {}
    for p in args.items:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                it = json.loads(l)
                items[it["pair_id"]] = it
    try:
        ann, skipped_blank = load_annotations(args.annotations, args.blank_as_uncertain, args.allow_incomplete)
        check_annotator_sets(ann, args.n_raters)
    except AnnotationError as e:
        sys.exit(f"ERROR: {e}")
    pairs, excluded = convert(items, ann, args.min_votes, args.n_raters, args.min_drift_raters)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".excluded.jsonl", "w", encoding="utf-8") as f:
        for r in excluded:
            f.write(json.dumps(r) + "\n")
    unannotated = sorted(set(items) - {p for p, rows in ann.items() if rows})
    print(f"{len(items)} items, {sum(1 for rows in ann.values() if rows)} annotated, {len(unannotated)} with no annotations")
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
    if skipped_blank:
        print(f"note: skipped {len(skipped_blank)} completely blank placeholder row(s)", file=sys.stderr)
    if LENIENT_BLANKS:
        print(f"warning: {len(LENIENT_BLANKS)} blank cell(s) treated as uncertain (--blank-as-uncertain), "
              f"e.g. {LENIENT_BLANKS[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
