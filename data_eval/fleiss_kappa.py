#!/usr/bin/env python3
"""Measure agreement among human annotators on preference pairs and drift labels.

Usage (from repository root):
    python -m data_eval.fleiss_kappa examples/labeling/sample_annotations.csv \
        --output-dir reports/labeling_pilot

Requires Python 3.9+ and no third-party packages.

Expected CSV columns (one row per annotator per pair):
    pair_id,annotator_id,preference,a_drift_types,b_drift_types

The A/B names are fixed candidate identities, NOT chosen/rejected labels.
Three annotators independently assess the same pairs (default --num-raters 3).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

DRIFT_TYPES = (
    "unrelated_edit",
    "scope_creep",
    "loop",
    "edits_tests",
    "hardcode",
    "skipped_requirement",
)
PREFERENCES = ("A", "B", "tie")
REQUIRED_COLUMNS = (
    "pair_id",
    "annotator_id",
    "preference",
    "a_drift_types",
    "b_drift_types",
)


class AnnotationError(ValueError):
    """Bad annotation input that should be corrected, not silently ignored."""


def parse_drift_types(value: str, *, location: str) -> frozenset[str] | None:
    """Return a set of drift labels, empty set for 'none', None for 'uncertain'.

    Multi-label entries can use semicolon or vertical-bar separators. None and
    uncertain must occur alone; empty cells are never implicitly 'none'.
    """
    if not isinstance(value, str) or not value.strip():
        raise AnnotationError(f"{location}: drift label is empty; use none or uncertain explicitly")
    text = value.strip().lower().replace("|", ";")
    parts = [part.strip() for part in text.split(";")]
    if any(not part for part in parts):
        raise AnnotationError(f"{location}: empty label between separators")
    if len(parts) != len(set(parts)):
        raise AnnotationError(f"{location}: duplicate drift label")
    allowed = set(DRIFT_TYPES) | {"none", "uncertain"}
    unknown = sorted(set(parts) - allowed)
    if unknown:
        raise AnnotationError(f"{location}: unknown drift label(s): {', '.join(unknown)}")
    if len(parts) > 1 and ("none" in parts or "uncertain" in parts):
        raise AnnotationError(f"{location}: none/uncertain cannot coexist with another label")
    if parts == ["uncertain"]:
        return None
    if parts == ["none"]:
        return frozenset()
    return frozenset(parts)


def load_annotations(paths: Path | Sequence[Path], *, num_raters: int = 3) -> dict[str, dict[str, dict[str, Any]]]:
    """Read and validate a complete independent-labeling CSV.

    Every pair must have the same num_raters annotators, and the set of IDs must
    be identical for every pair. Uncertain ratings are permitted but explicit.
    """
    if num_raters < 2:
        raise AnnotationError("num_raters must be at least 2")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if isinstance(paths, Path):
        paths = [paths]
    if not paths:
        raise AnnotationError("Provide at least one CSV file")
    for path in paths:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise AnnotationError("CSV is empty or has no header")
                fields = [name.strip() for name in reader.fieldnames]
                if len(fields) != len(set(fields)):
                    raise AnnotationError("CSV has duplicate column names")
                missing = set(REQUIRED_COLUMNS) - set(fields)
                if missing:
                    raise AnnotationError("CSV missing columns: " + ", ".join(sorted(missing)))
                for line_number, row in enumerate(reader, start=2):
                    if None in row:
                        raise AnnotationError(f"row {line_number}: too many CSV fields; quote commas within values")
                    # Match trimmed header labels (without discarding extra metadata columns).
                    row = {key.strip(): val for key, val in row.items()}
                    pair_id = (row["pair_id"] or "").strip()
                    annotator_id = (row["annotator_id"] or "").strip()
                    if not pair_id or not annotator_id:
                        raise AnnotationError(f"row {line_number}: pair_id and annotator_id are required")
                    if annotator_id in grouped[pair_id]:
                        raise AnnotationError(f"row {line_number}: duplicate annotation for pair {pair_id!r}, annotator {annotator_id!r}")
                    pref = (row["preference"] or "").strip()
                    if pref.lower() == "uncertain":
                        pref = "uncertain"
                    elif pref.upper() in ("A", "B"):
                        pref = pref.upper()
                    elif pref.lower() == "tie":
                        pref = "tie"
                    else:
                        raise AnnotationError(
                            f"row {line_number}: preference {pref!r} must be A, B, tie, or uncertain"
                        )
                    grouped[pair_id][annotator_id] = {
                        "preference": pref,
                        "A": parse_drift_types(row["a_drift_types"], location=f"row {line_number} a_drift_types"),
                        "B": parse_drift_types(row["b_drift_types"], location=f"row {line_number} b_drift_types"),
                    }
        except (OSError, UnicodeError, csv.Error) as exc:
            raise AnnotationError(f"Cannot read annotations CSV {path}: {exc}") from exc
    if not grouped:
        raise AnnotationError("No annotations found")
    canonical_raters: set[str] | None = None
    for pair_id, responses in sorted(grouped.items()):
        raters = set(responses)
        if len(raters) != num_raters:
            raise AnnotationError(
                f"pair {pair_id!r} has {len(raters)} distinct annotator(s); expected {num_raters}"
            )
        if canonical_raters is None:
            canonical_raters = raters
        elif raters != canonical_raters:
            raise AnnotationError(
                f"pair {pair_id!r}: different annotators; all pairs must be labeled by the same team"
            )
    return dict(grouped)


def fleiss_kappa(units: Sequence[Sequence[str]], categories: Sequence[str], *, num_raters: int) -> dict[str, Any]:
    """Compute standard, nominal, fixed-rater-count Fleiss' kappa.

    Units are independently rated items. Categories are mutually exclusive for
    THIS metric. Multi-label drift is evaluated as six separate binary metrics.
    Uncertain items must be excluded by caller (only complete units passed).
    """
    category_set = set(categories)
    if not units:
        return {
            "items": 0,
            "raters_per_item": num_raters,
            "kappa": None,
            "observed_agreement": None,
            "chance_agreement": None,
            "unanimous_fraction": None,
            "category_prevalence": {c: None for c in categories},
            "undefined_reason": "No completely labeled items for this metric",
        }
    if len(categories) < 2 or len(category_set) != len(categories):
        raise ValueError("At least two unique categories are required")
    total_counts: Counter[str] = Counter()
    observed_sum = 0.0
    unanimous = 0
    for idx, ratings in enumerate(units):
        if len(ratings) != num_raters:
            raise ValueError(f"item {idx}: expected {num_raters} ratings, found {len(ratings)}")
        if any(label not in category_set for label in ratings):
            raise ValueError(f"item {idx}: unknown rating")
        counts = Counter(ratings)
        total_counts.update(counts)
        observed_sum += sum(n * (n - 1) for n in counts.values()) / (num_raters * (num_raters - 1))
        if max(counts.values()) == num_raters:
            unanimous += 1
    n_items = len(units)
    prevalence = {c: total_counts[c] / (n_items * num_raters) for c in categories}
    p_observed = observed_sum / n_items
    p_chance = sum(p * p for p in prevalence.values())
    if abs(1.0 - p_chance) < 1e-12:
        kappa = None
        reason = "All ratings use one category; chance agreement is 100%, so kappa is undefined"
    else:
        kappa = (p_observed - p_chance) / (1.0 - p_chance)
        reason = None
    return {
        "items": n_items,
        "raters_per_item": num_raters,
        "kappa": kappa,
        "observed_agreement": p_observed,
        "chance_agreement": p_chance,
        "unanimous_fraction": unanimous / n_items,
        "category_prevalence": prevalence,
        "undefined_reason": reason,
    }


def _detail_row(pair_id: str, dimension: str, raters: list[str], values: list[str]) -> dict[str, str]:
    c = Counter(values)
    majority, freq = c.most_common(1)[0]
    return {
        "pair_id": pair_id,
        "dimension": dimension,
        "ratings": "; ".join(f"{r}:{v}" for r, v in zip(raters, values)),
        "majority_label": majority if freq > len(values) // 2 and majority != "uncertain" else "",
        "has_uncertain": "yes" if "uncertain" in values else "no",
    }


def analyze_annotations(grouped: dict[str, dict[str, dict[str, Any]]], *, num_raters: int = 3) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return machine-readable agreement summary and rows needing discussion."""
    preference_units: list[list[str]] = []
    binary_units: dict[str, list[list[str]]] = {label: [] for label in (*DRIFT_TYPES, "any_drift")}
    skip_prefs: list[str] = []
    skip_candidates: list[str] = []
    disputes: list[dict[str, str]] = []
    total_candidate_units = 0
    for pair_id, responses in sorted(grouped.items()):
        raters = sorted(responses)
        if len(raters) != num_raters:
            raise AnnotationError(f"pair {pair_id!r}: expected {num_raters} raters")
        prefs = [responses[r]["preference"] for r in raters]
        if "uncertain" in prefs:
            skip_prefs.append(pair_id)
        else:
            preference_units.append(prefs)
        if len(set(prefs)) != 1 or "uncertain" in prefs:
            disputes.append(_detail_row(pair_id, "preference", raters, prefs))
        for candidate in ("A", "B"):
            total_candidate_units += 1
            labels = [responses[r][candidate] for r in raters]
            if any(label is None for label in labels):
                skip_candidates.append(f"{pair_id}:{candidate}")
                # Show the skipped item even if all raters said uncertain.
                disputes.append(_detail_row(pair_id, f"{candidate}:drift_label_status", raters,
                                            ["uncertain" if label is None else "labeled" for label in labels]))
                continue
            for drift_type in DRIFT_TYPES:
                votes = ["yes" if drift_type in lab else "no" for lab in labels]
                binary_units[drift_type].append(votes)
                if len(set(votes)) != 1:
                    disputes.append(_detail_row(pair_id, f"{candidate}:{drift_type}", raters, votes))
            votes = ["yes" if lab else "no" for lab in labels]
            binary_units["any_drift"].append(votes)
            if len(set(votes)) != 1:
                disputes.append(_detail_row(pair_id, f"{candidate}:any_drift", raters, votes))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_pairs": len(grouped),
        "num_candidate_actions": total_candidate_units,
        "num_raters": num_raters,
        "annotator_ids": sorted(next(iter(grouped.values())).keys()),
        "preference": {
            **fleiss_kappa(preference_units, PREFERENCES, num_raters=num_raters),
            "excluded_pair_ids": skip_prefs,
        },
        "drift": {
            label: fleiss_kappa(binary_units[label], ("no", "yes"), num_raters=num_raters)
            for label in (*DRIFT_TYPES, "any_drift")
        },
        "excluded_candidate_ids_for_drift": skip_candidates,
        "items_requiring_review": len({row["pair_id"] for row in disputes}),
        "disagreement_rows": len(disputes),
        "notes": [
            "A and B are fixed candidate identities, not chosen/rejected labels.",
            "Preference is nominal A, B, or tie; uncertain is excluded from preference kappa.",
            "Drift is multi-label, evaluated as six separate yes/no kappas on candidate actions.",
            "An uncertain drift rating excludes that candidate from ALL drift metrics; never treat it as none.",
            "any_drift means at least one of the six drift labels; it does not measure correctness.",
            "Undefined kappa (null) means no eligible items or zero category variation; consult observed agreement.",
            "Disagreements and majority votes are diagnostics, not finalized gold labels; adjudicate separately.",
        ],
    }
    return report, disputes


def save_results(report: dict[str, Any], disputes: list[dict[str, str]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "kappa_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "review_disagreements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "dimension", "ratings", "majority_label", "has_uncertain"])
        writer.writeheader()
        writer.writerows(disputes)


def _fmt(v: float | None) -> str:
    return "undefined" if v is None else f"{v:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", type=Path, nargs="+", help="One combined CSV or separate CSVs from each human annotator")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/kappa"), help="Where to save report and disagreements CSV")
    parser.add_argument("--num-raters", type=int, default=3, help="Expected fixed annotator count (default 3)")
    args = parser.parse_args(argv)
    try:
        grouped = load_annotations(args.annotations, num_raters=args.num_raters)
        report, disputes = analyze_annotations(grouped, num_raters=args.num_raters)
        save_results(report, disputes, args.output_dir)
    except (AnnotationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Loaded {report['num_pairs']} pairs, {report['num_raters']} raters, {report['num_candidate_actions']} candidate actions")
    pref = report["preference"]
    print(f"Preference (A/B/tie): Fleiss kappa {_fmt(pref['kappa'])}; "+
          f"raw agreement {_fmt(pref['observed_agreement'])}; eligible pairs {pref['items']}/{report['num_pairs']}")
    print("Drift (each candidate labeled yes/no for each type):")
    for label, metric in report["drift"].items():
        print(f"  {label:22s} kappa={_fmt(metric['kappa']):>9s}  "+
              f"agreement={_fmt(metric['observed_agreement']):>9s}  items={metric['items']}")
    if pref["excluded_pair_ids"] or report["excluded_candidate_ids_for_drift"]:
        print(f"Excluded for uncertain: {len(pref['excluded_pair_ids'])} preference pairs, "+
              f"{len(report['excluded_candidate_ids_for_drift'])} candidate actions")
    print(f"Items needing review: {report['items_requiring_review']} pair(s); {len(disputes)} disagreement/uncertainty rows")
    print(f"Saved {args.output_dir / 'kappa_report.json'} and {args.output_dir / 'review_disagreements.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
