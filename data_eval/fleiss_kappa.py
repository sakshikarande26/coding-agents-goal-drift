#!/usr/bin/env python3
"""Fleiss' kappa for candidate-pair annotations (16-column format).

Run as a script:
    python fleiss_kappa.py rater1.csv rater2.csv rater3.csv --output-dir reports/kappa
Or from the repository root if this file lives in data_eval/:
    python -m data_eval.fleiss_kappa rater1.csv rater2.csv rater3.csv --output-dir reports/kappa

Python 3.9+; Python standard library only.

Each row: pair_id, annotator_id, preference (A/B/tie/uncertain), 6 a_* drift
columns and 6 b_* drift columns (yes/no/uncertain). An optional notes
column and any other metadata columns are ignored by the statistic.

The same, fixed A/B candidate identities are used for all drift labels.
The agreed labeling rules govern HUMAN decisions, not a transformation of
labels in this analysis script. Preference and six drift ratings are
independent. Unknown and blank values cannot silently become 'no'.
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
DRIFT_COLUMNS = tuple(f"{p}_{label}" for p in ("a", "b") for label in DRIFT_TYPES)
LABEL_COLUMNS = ("preference",) + DRIFT_COLUMNS
REQUIRED_COLUMNS = ("pair_id", "annotator_id") + LABEL_COLUMNS


class AnnotationError(ValueError):
    """Input problems requiring correction rather than silent reinterpretation."""


def parse_preference(value: str, *, location: str) -> str:
    val = (value or "").strip()
    if val.upper() in ("A", "B"):
        return val.upper()
    if val.lower() in ("tie", "uncertain"):
        return val.lower()
    raise AnnotationError(f"{location}: preference {val!r} must be A, B, tie, or uncertain")


def parse_drift(value: str, *, location: str) -> str:
    val = (value or "").strip().lower()
    mapping = {
        "yes": "yes", "y": "yes", "true": "yes", "1": "yes",
        "no": "no", "n": "no", "false": "no", "0": "no",
        "uncertain": "uncertain",
    }
    if val not in mapping:
        raise AnnotationError(f"{location}: drift {val!r} must be yes, no, or uncertain (1/0 and true/false accepted)")
    return mapping[val]


def load_annotations(
    paths: Path | Sequence[Path], *, num_raters: int = 3,
    allow_incomplete: bool = False, audit: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load independently labeled pair items, grouping by stable pair_id.

    Default is strict: every pair needs all num_raters, and no empty labels.
    When allow_incomplete=True, entirely empty placeholder rows are ignored;
    pairs lacking all raters are excluded. Partially filled rows always error.
    Uncertain is an explicit label, not a blank or a placeholder.
    """
    if num_raters < 2:
        raise AnnotationError("num_raters must be at least 2")
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    paths = [Path(path) for path in paths]
    if not paths:
        raise AnnotationError("Provide at least one CSV file")
    all_pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seen: set[tuple[str, str]] = set()
    blank_rows: list[str] = []
    for path in paths:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise AnnotationError(f"{path}: CSV has no header")
                fields = [name.strip() for name in reader.fieldnames]
                if len(fields) != len(set(fields)):
                    raise AnnotationError(f"{path}: duplicate header column names")
                missing = sorted(set(REQUIRED_COLUMNS) - set(fields))
                if missing:
                    raise AnnotationError(f"{path}: CSV missing columns: {', '.join(missing)}")
                for line_number, raw in enumerate(reader, start=2):
                    location = f"{path}:{line_number}"
                    if None in raw:
                        raise AnnotationError(f"{location}: too many comma-separated fields")
                    row = {(key or "").strip(): value for key, value in raw.items()}
                    pair = (row["pair_id"] or "").strip()
                    rater = (row["annotator_id"] or "").strip()
                    if not pair or not rater:
                        raise AnnotationError(f"{location}: pair_id and annotator_id must be nonempty")
                    key = (pair, rater)
                    if key in seen:
                        raise AnnotationError(f"{location}: duplicate annotation for pair {pair!r}, annotator {rater!r}")
                    seen.add(key)
                    all_pairs.setdefault(pair, {})
                    empties = [col for col in LABEL_COLUMNS if not (row[col] or "").strip()]
                    if empties:
                        if allow_incomplete and len(empties) == len(LABEL_COLUMNS):
                            blank_rows.append(f"{pair}:{rater}")
                            continue
                        if len(empties) == len(LABEL_COLUMNS):
                            raise AnnotationError(
                                f"{location}: completely unlabeled placeholder row; "
                                "complete it or use --allow-incomplete to exclude it"
                            )
                        raise AnnotationError(f"{location}: partially labeled row: empty {', '.join(empties)}; "
                                              "enter yes/no/uncertain explicitly")
                    all_pairs[pair][rater] = {
                        "preference": parse_preference(row["preference"], location=f"{location} preference"),
                        "A": {label: parse_drift(row[f"a_{label}"], location=f"{location} a_{label}")
                              for label in DRIFT_TYPES},
                        "B": {label: parse_drift(row[f"b_{label}"], location=f"{location} b_{label}")
                              for label in DRIFT_TYPES},
                    }
        except (OSError, UnicodeError, csv.Error) as exc:
            raise AnnotationError(f"Cannot read {path}: {exc}") from exc

    if not seen:
        raise AnnotationError("No annotation rows found")
    eligible: dict[str, dict[str, dict[str, Any]]] = {}
    excluded: list[str] = []
    canonical_raters: set[str] | None = None
    for pair, responses in sorted(all_pairs.items()):
        raters = set(responses)
        if len(raters) != num_raters:
            if not allow_incomplete:
                raise AnnotationError(f"pair {pair!r}: found {len(raters)} annotators, expected {num_raters}")
            excluded.append(pair)
            continue
        if canonical_raters is None:
            canonical_raters = raters
        elif raters != canonical_raters:
            raise AnnotationError(
                f"pair {pair!r}: annotator set differs from other complete pairs; "
                "same annotators must label every eligible pair"
            )
        eligible[pair] = responses
    if not eligible:
        raise AnnotationError(
            f"No pairs have all {num_raters} annotators; "
            "provide additional reviewers or change --num-raters for a diagnostic run"
        )
    if audit is not None:
        audit.clear()
        audit.update({
            "source_pairs": len(all_pairs),
            "source_csv_files": [str(p) for p in paths],
            "blank_placeholder_rows_ignored": len(blank_rows),
            "blank_placeholder_ids": sorted(blank_rows),
            "pairs_excluded_for_missing_raters": len(excluded),
            "excluded_incomplete_pair_ids": excluded,
            "allow_incomplete": allow_incomplete,
        })
    return eligible


def fleiss_kappa(
    units: Sequence[Sequence[str]], categories: Sequence[str], *, num_raters: int,
) -> dict[str, Any]:
    """Standard unweighted nominal Fleiss' kappa for fixed n raters/item.

    All uncertain-containing units must be removed before this function.
    It also operates for n=2 (diagnostic only; n=3 for the intended pilot).
    """
    if num_raters < 2:
        raise ValueError("num_raters must be >= 2")
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("At least two unique categories are required")
    if not units:
        return {
            "items": 0, "raters_per_item": num_raters, "kappa": None,
            "observed_agreement": None, "chance_agreement": None,
            "unanimous_fraction": None,
            "category_prevalence": {category: None for category in categories},
            "undefined_reason": "No eligible fully rated items",
        }
    counts_all: Counter[str] = Counter()
    agreement_sum = 0.0
    unanimous = 0
    allowed = set(categories)
    for idx, ratings in enumerate(units):
        if len(ratings) != num_raters:
            raise ValueError(f"item {idx}: expected {num_raters} ratings, got {len(ratings)}")
        if any(val not in allowed for val in ratings):
            raise ValueError(f"item {idx}: unknown category")
        counts = Counter(ratings)
        counts_all.update(counts)
        agreement_sum += sum(n * (n - 1) for n in counts.values()) / (num_raters * (num_raters - 1))
        unanimous += int(max(counts.values()) == num_raters)
    n_items = len(units)
    prevalence = {c: counts_all[c] / (n_items * num_raters) for c in categories}
    observed = agreement_sum / n_items
    chance = sum(p * p for p in prevalence.values())
    if abs(1.0 - chance) < 1e-12:
        kappa = None
        reason = "All eligible ratings use one category; kappa is mathematically undefined"
    else:
        kappa = (observed - chance) / (1.0 - chance)
        reason = None
    return {
        "items": n_items,
        "raters_per_item": num_raters,
        "kappa": kappa,
        "observed_agreement": observed,
        "chance_agreement": chance,
        "unanimous_fraction": unanimous / n_items,
        "category_prevalence": prevalence,
        "undefined_reason": reason,
    }


def _detail_row(pair_id: str, dimension: str, raters: list[str], votes: list[str]) -> dict[str, str]:
    counts = Counter(votes)
    majority, count = counts.most_common(1)[0]
    return {
        "pair_id": pair_id,
        "dimension": dimension,
        "ratings": "; ".join(f"{r}:{v}" for r, v in zip(raters, votes)),
        "majority_label": majority if count > len(votes) // 2 and majority != "uncertain" else "",
        "has_uncertain": "yes" if "uncertain" in votes else "no",
    }


def analyze_annotations(
    grouped: dict[str, dict[str, dict[str, Any]]], *, num_raters: int = 3,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Summarize preference and per-drift-type kappa, plus review disagreements.

    Each (pair_id, candidate A/B) is an independent binary observation.
    A category is excluded only when a rater is uncertain about THAT category.
    any_drift requires all six categories for all raters to be known.
    """
    preference_units: list[list[str]] = []
    drift_units: dict[str, list[list[str]]] = {name: [] for name in (*DRIFT_TYPES, "any_drift")}
    excluded_pref: list[str] = []
    excluded_by_category: dict[str, list[str]] = {name: [] for name in (*DRIFT_TYPES, "any_drift")}
    disputes: list[dict[str, str]] = []
    num_candidates = 0
    for pair_id, responses in sorted(grouped.items()):
        raters = sorted(responses)
        if len(raters) != num_raters:
            raise AnnotationError(f"pair {pair_id!r}: expected {num_raters} raters")
        prefs = [responses[r]["preference"] for r in raters]
        if "uncertain" in prefs:
            excluded_pref.append(pair_id)
        else:
            preference_units.append(prefs)
        if len(set(prefs)) != 1 or "uncertain" in prefs:
            disputes.append(_detail_row(pair_id, "preference", raters, prefs))
        for candidate in ("A", "B"):
            num_candidates += 1
            candidate_id = f"{pair_id}:{candidate}"
            for drift_type in DRIFT_TYPES:
                votes = [responses[r][candidate][drift_type] for r in raters]
                if "uncertain" in votes:
                    excluded_by_category[drift_type].append(candidate_id)
                else:
                    drift_units[drift_type].append(votes)
                if len(set(votes)) != 1 or "uncertain" in votes:
                    disputes.append(_detail_row(pair_id, f"{candidate}:{drift_type}", raters, votes))
            all_known = all(
                responses[r][candidate][type_name] != "uncertain"
                for r in raters for type_name in DRIFT_TYPES
            )
            if not all_known:
                excluded_by_category["any_drift"].append(candidate_id)
            else:
                votes = ["yes" if any(responses[r][candidate][typ] == "yes" for typ in DRIFT_TYPES)
                         else "no" for r in raters]
                drift_units["any_drift"].append(votes)
                if len(set(votes)) != 1:
                    disputes.append(_detail_row(pair_id, f"{candidate}:any_drift", raters, votes))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_pairs": len(grouped),
        "num_candidate_actions": num_candidates,
        "num_raters": num_raters,
        "annotator_ids": sorted(next(iter(grouped.values())).keys()),
        "preference": {
            **fleiss_kappa(preference_units, PREFERENCES, num_raters=num_raters),
            "excluded_pair_ids": excluded_pref,
            "excluded_count": len(excluded_pref),
        },
        "drift": {
            typ: {
                **fleiss_kappa(drift_units[typ], ("no", "yes"), num_raters=num_raters),
                "excluded_count": len(excluded_by_category[typ]),
                "excluded_candidate_ids": excluded_by_category[typ],
            } for typ in (*DRIFT_TYPES, "any_drift")
        },
        "excluded_candidate_ids_by_drift": excluded_by_category,
        "excluded_candidate_ids_for_drift": excluded_by_category["any_drift"],
        "items_requiring_review": len({r["pair_id"] for r in disputes}),
        "disagreement_rows": len(disputes),
        "notes": [
            "A/B identify fixed candidate actions, independent of the preference selection.",
            "Preference kappa uses A/B/tie; excludes pairs with ANY uncertain preference.",
            "Each of six binary drift kappas is calculated over A+B candidate actions independently.",
            "Uncertainty excludes only that candidate/category observation from that category's kappa.",
            "any_drift is a stricter optional diagnostic, excluding candidates with ANY uncertain category.",
            "Missing cells are never interpreted as no; blank placeholders are only skipped by explicit flag.",
            "No automatic loop, hardcode, or skipped_requirement inference is performed: human rubric controls labels.",
            "Kappa measures independent rater agreement; majority shown in review report is NOT an adjudication.",
            "Null kappa indicates no eligible items or no prevalence variation; see raw agreement and reason.",
        ],
    }
    return report, disputes


def save_results(report: dict[str, Any], disputes: list[dict[str, str]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "kappa_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "review_disagreements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "dimension", "ratings", "majority_label", "has_uncertain"])
        writer.writeheader()
        writer.writerows(disputes)


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", type=Path, nargs="+", help="Separate reviewer CSV files or one combined CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/kappa"))
    parser.add_argument("--num-raters", type=int, default=3, help="Expected number of independent annotators (default: 3)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="For a diagnostic run, ignore wholly blank placeholder rows and pairs missing reviewers")
    args = parser.parse_args(argv)
    try:
        audit: dict[str, Any] = {}
        grouped = load_annotations(args.annotations, num_raters=args.num_raters,
                                   allow_incomplete=args.allow_incomplete, audit=audit)
        report, disputes = analyze_annotations(grouped, num_raters=args.num_raters)
        report["input_validation"] = audit
        save_results(report, disputes, args.output_dir)
    except (AnnotationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Analyzed {report['num_pairs']} pairs ({audit['source_pairs']} in CSVs), "
          f"{report['num_raters']} reviewers, {report['num_candidate_actions']} candidate actions")
    if audit["pairs_excluded_for_missing_raters"]:
        print(f"Excluded {audit['pairs_excluded_for_missing_raters']} incomplete pairs; "
              f"ignored {audit['blank_placeholder_rows_ignored']} entirely blank rows")
    pref = report["preference"]
    print(f"Preference A/B/tie: kappa={_fmt(pref['kappa'])}, agreement={_fmt(pref['observed_agreement'])}, "
          f"eligible={pref['items']}/{report['num_pairs']}, excluded_uncertain={pref['excluded_count']}")
    print("Drift: binary yes/no per candidate, category-specific uncertain exclusions:")
    for typ, metric in report["drift"].items():
        print(f"  {typ:21s} kappa={_fmt(metric['kappa']):>9s} "
              f"agreement={_fmt(metric['observed_agreement']):>9s} "
              f"eligible={metric['items']:>3d}/{report['num_candidate_actions']:<3d} "
              f"excluded_uncertain={metric['excluded_count']}")
    print(f"Review needed: {report['items_requiring_review']} distinct pairs; {len(disputes)} issue rows")
    print(f"Saved {args.output_dir / 'kappa_report.json'} and "
          f"{args.output_dir / 'review_disagreements.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
