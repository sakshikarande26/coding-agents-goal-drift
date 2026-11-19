# Track 2: Fleiss' kappa for preference and goal-drift annotations

This is a **standalone evaluation script**, not a reranker and not a trainer. It checks **how consistently three people apply the labeling guide** to the same two-candidate comparisons. It does *not* determine which labels are correct or generate labeled pairs from Khoa's candidate JSONL.

## What the team labels

For each fixed two-candidate item, show all three annotators the **same task description**, **same shared prefix**, and **same Candidate A and Candidate B**. Keep A and B in the same order for everyone; do not call one "chosen" or "rejected" before annotation. Each person independently records:

1. **Preference**: `A`, `B`, `tie`, or `uncertain` (can't judge from provided evidence).
2. **Drift labels for Candidate A** (any subset of the six types, `none`, or `uncertain`).
3. **Drift labels for Candidate B** (same rules).

Allowed drift types: `unrelated_edit`, `scope_creep`, `loop`, `edits_tests`, `hardcode`, `skipped_requirement`.

Use a semicolon to separate multiple labels, e.g. `loop;scope_creep`. Use **`none` alone** only when the action doesn't exhibit any of the six types. Use **`uncertain` alone** when there is not enough evidence. `none` is *not* the same as uncertain and does *not* mean the action is preferred. Don't mark drift simply because the action is suboptimal; preference is a separate judgment.

Some labels, especially `skipped_requirement` and `hardcode`, may require more context than an action alone provides. Mark uncertain if the evidence is insufficient.

## Inputs

The script accepts either one combined CSV with a row per (pair, annotator), or **three separate CSV files** (one from each annotator). Required columns:

| Field | Meaning | Example |
|---|---|---|
| `pair_id` | Unique, stable identifier for the fixed comparison | `pair_004` |
| `annotator_id` | Consistent rater ID across all comparisons | `rater_2` |
| `preference` | A, B, tie, or uncertain | `A` |
| `a_drift_types` | Semicolon-separated drift labels for A, none, or uncertain | `none` |
| `b_drift_types` | Semicolon-separated drift labels for B, none, or uncertain | `loop;scope_creep` |

Example CSV rows:

```csv
pair_id,annotator_id,preference,a_drift_types,b_drift_types
pair_001,rater_1,A,none,loop
pair_001,rater_2,A,none,loop
pair_001,rater_3,B,none,loop;scope_creep
```

A row for each of the three raters must exist for *every* pair. The script rejects incomplete data, repeated pair/rater IDs, unknown drift categories and inconsistent rater sets. **Blank annotations are not accepted**: complete the sheet first, and use `uncertain` when appropriate.

The ZIP includes an 8-pair **synthetic** demonstration dataset (`sample_annotations.csv`) plus three separate filled demo sheets (`rater_1_example.csv`, etc.) and three blank individual sheets (`rater_1_blank.csv`, etc.). Replace example pair IDs with the IDs Khoa supplies. Each teammate should label their sheet **independently before discussing disagreements**.

### Run it from the Git repository root

For one merged CSV:

```bash
python -m data_eval.fleiss_kappa examples/labeling/sample_annotations.csv --output-dir reports/kappa_pilot
```

For three independently completed sheets:

```bash
python -m data_eval.fleiss_kappa rater_1.csv rater_2.csv rater_3.csv --output-dir reports/kappa_pilot
```

Python 3.9+; standard library only. To run the tests:

```bash
python -m unittest discover -s tests -v
```

## What it calculates

### A. **Preference agreement** (one three-category Fleiss' kappa)

Each pair is one item. The three raters choose among A, B, and tie. For example, if two choose A and one chooses B, Fleiss' pair-level observed agreement is 1/3 (one of the three rater-pairs agrees). If all three choose A, agreement is 1. If the choices are A, B and tie, agreement is 0.

Fleiss' kappa then adjusts the average agreement for the agreement expected by chance based on how often each label is used:

- \(P_i = \sum_c n_{ic}(n_{ic}-1)/(r(r-1))\)
- \(\bar P = \frac{1}{N}\sum_i P_i\)
- \(p_c = \frac{\sum_i n_{ic}}{Nr}\), \(P_e = \sum_c p_c^2\)
- \(\kappa = (\bar P - P_e)/(1-P_e)\)

Here \(N\) is the number of eligible pairs, \(r=3\) raters, and \(c\) is the category.

### B. **Drift-label agreement** (six separate binary Fleiss' kappas)

A candidate may exhibit **multiple drift types**, so we **must not** treat the whole set of labels as one mutually exclusive category. Instead, **each candidate** (A and B separately) is one item for **each** of the six categories, rated `yes`/`no` by each person. For 20 pairs with no uncertainty, that's 40 candidate items per drift type, each with 3 ratings. The script also reports a binary `any_drift` agreement measure.

For instance, raters' drift labels on B are:

- R1: `loop`
- R2: `loop;scope_creep`
- R3: `none`

For **loop**, the binary votes are `[yes, yes, no]`; for **scope_creep**, `[no, yes, no]`; for **edits_tests**, `[no, no, no]`; and similarly for the other three categories.

### C. How `uncertain` is handled

- A preference `uncertain` from **any** rater excludes that *pair* from **preference kappa**, but drift judgments for its candidates can still be used.
- A candidate's drift labels marked `uncertain` by **any** rater exclude that *candidate* from **all drift kappas**, but preference for its pair can still be used.
- The script reports every exclusion and flags it for review. It does **not** silently convert uncertainty to `none` or reduce the number of raters to 2.

## Output

`reports/kappa_pilot/kappa_report.json`: preference kappa, seven binary drift kappas (six categories plus any drift), raw observed agreement, chance agreement, unanimous fraction, label prevalence, item counts, exclusions, and diagnostic notes.

`reports/kappa_pilot/review_disagreements.csv`: which pair and dimension had disagreements or uncertainty, the three individual ratings, and any simple majority. **A majority is not a finalized gold label**; review disagreements together only after independent annotation, record an adjudicated consensus separately.

The command line prints a compact summary.

## Interpretation and limitations

- `κ = 1`: perfect agreement beyond chance; `κ ≈ 0`: no better than chance, relative to observed label prevalence; negative kappa: worse than chance. These are **agreement**, not accuracy, scores.
- If all raters assign the same category across all eligible items (e.g., no one flags `hardcode`), expected chance agreement is 1, making kappa **mathematically undefined**. The script reports `null`/`undefined`, but still reports raw agreement and prevalence. This is expected for rare drift categories.
- With a tiny 20-pair pilot and rare labels, category-specific kappa estimates can be unstable. Always report eligible item count, prevalence and raw agreement alongside kappa, and review disagreement examples rather than using a single arbitrary cutoff.
- This script doesn't validate Khoa's raw `prompt_messages`, doesn't check the source candidate pools, and doesn't generate DPO `chosen`/`rejected` records. Your current `validate_pairs.py` still expects a **single string** `drift_type`; **the multilabel human-annotation format here is a separate, provisional analysis format**. Agree on and update the final pair schema before exporting consensus training pairs.
- The action presented to raters should have only the information the agent could have had at that decision point. Do not inadvertently show future rollout outcomes when comparing proposed actions; if the label needs future evidence, mark uncertain or define a separate retrospective labeling workflow.
