# Analyze the archived results without models

Python 3.10 or newer is sufficient. The analysis package uses only the Python
standard library. It does not import PyTorch, load models, run inference, fit a
predictor, download anything, or contact a service.

Verified locally with Windows Python 3.11 and WSL Python 3.10 using `-S`
(site packages disabled): all 15 public tests and the complete saved-result
validator passed on both runtimes. Python 3.12 is included in the optional CI
template; no active CI run is claimed here.

From the repository root:

```sh
python -m analysis.analyze_results validate --repo-root .
python -m unittest discover -s tests_public -v
python -m analysis.analyze_results analyze --repo-root . --output analysis_output
```

`--output-dir` is an alias for `--output`. Choose a new or empty output directory;
the tool refuses the repository root, its ancestors, and locations inside frozen
`data/`, `results/`, or `publication/`. Existing output files are never overwritten.
`validate` prints a compact result and exits zero only when all checks pass.
`analyze` writes CSV/JSON and uses the same exit status; a failed audit is clearly
marked in its JSON rather than repaired or silently omitted.

## Inputs and identity

The tool expects:

```text
data/a1/<scenario>/{samples.jsonl,test_gold.jsonl,benchmark_lock.json,manifest.json}
results/matched_budget/final_summary.json
results/matched_budget/cases/<case_id>/case.json
results/matched_budget/cases/<case_id>/acceptance.json
results/matched_budget/cases/<case_id>/result.json
results/matched_budget/cases/<case_id>/history.json
results/matched_budget/cases/<case_id>/selection_before_test.json
results/matched_budget/cases/<case_id>/test_predictions_correct.jsonl
results/matched_budget/cases/<case_id>/test_C1_answer_nll.jsonl
```

The two scenarios are checked separately. A sample ID can occur in both scenarios;
the effective identity is **(scenario, sample ID)**, not sample ID alone. Frozen
sample/gold byte fingerprints, split membership/counts, prediction order, complete
coverage, original unrounded gold, and source case identity are verified.

This release contains 36 planned cases: **30 historically accepted, one
quarantined attempt, and five missing**. The 30 accepted cases contain 8,910 saved
predictions. The quarantined pooling attempt has another 152 saved predictions;
its values are reconstructed separately and never fill the official pooling
table. `result.status == "complete"` is not sufficient: each attempted case must
have an independently projected `acceptance.json` boolean
`valid_for_acceptance`. The historical rejection remains false even if the
saved numbers are internally consistent. Missing or false acceptance cannot be
replaced by a result file's own success claim.

The projected acceptance record preserves the original validator SHA and reason
when available. This analysis verifies the projection's identity and exclusion
rules; it does not authenticate an omitted original private file or re-execute
the historical contract.

## What is recomputed

The original strict parser accepts exactly one JSON object with one finite
numeric `pKD` field. Duplicate keys, booleans, NaN/Infinity, code fences, extra
fields, and explanatory prose are rejected. Finite values are not clipped.
Extreme malformed integer/recursion inputs fail safely. Every archived parse
record is independently checked against its raw text.

The analysis recomputes MAE and Spearman with average ranks for ties. Constant
or insufficient vectors retain undefined Spearman as null. Exact-target groups
and macro MAE are also reconstructed without calling the original NumPy code.
Any failed answer makes full-sample MAE null; the valid-only denominator and
failure counts remain separate. It does not change samples, labels, failure
rules, or the primary metric.

It verifies three recorded epochs, the recorded 1,548 updates, and the earliest
minimum **full validation answer-token NLL** selection. This is a consistency
audit of saved records, not proof obtained from rerunning training.

The seven fixed contrasts per scenario are:

| Contrast | Direction |
|---|---|
| depth_real | early-real minus mid-real |
| depth_neutral | early-neutral minus mid-neutral |
| content_early | early-neutral minus early-real |
| content_mid | mid-neutral minus mid-real |
| depth_content_difference_in_differences | depth_real minus depth_neutral |
| gate_zero_initialization | mid near-unit minus mid zero-initialized |
| balanced_pool_allocation | mid global16 minus mid balanced8+8 |

All seeds 17/42/2026 are explicitly retained. Mean and sample SD require **3/3**
finite seed values; any missing/invalid value leaves the summary null. Sample SD
is not a confidence interval. The tool does not resample bootstrap intervals,
add p-values, invent missing pooling runs, or select the best seed.

Stored C1 is checked as a row-mean difference in teacher-forced full-answer NLL.
Format/numeric/mixed category contributions use each row's full-answer token
denominator. Original reduction residuals are retained, rather than forcing
different floating-point reductions to agree. C1 is a computational branch
effect, not molecular information or biological causal validation. Raw token IDs
are retained, but this stdlib tool does not re-decode them with a model tokenizer.

## Outputs

- `analysis.json`: detailed independent checks, input SHA256, all attempted raw
  metrics, official/quarantined/missing IDs, seed tables and fixed contrasts.
- `cases.csv`: attempted raw metrics, with a separate official-inclusion flag;
  quarantined official MAE cells remain blank.
- `seed_summaries.csv`: explicit per-seed values and 3/3-only mean/sample SD.
- `paired_contrasts.csv`: the original five core plus gate/pooling contrasts,
  including null/0-of-3 pooling coverage.

The arithmetic tolerance of 1e-12 only compares independently reconstructed
floating-point numbers. It is not a new scientific superiority threshold.

## Historical execution is not a portable training runner

Historical training code and contracts document the experiment's provenance.
They refer to model caches, an original runtime, and a **2026-10-16 23:20 +08:00**
execution deadline. They are not a portable command for restarting that expired
campaign. This analysis does not revalidate an old execution window against
today or promote a previously rejected run. A future model reproduction needs
a new explicit environment and execution contract; saved-result recomputation
needs neither.

## Optional CI template

`ci/validate.yml.example` is intentionally **not** an active GitHub workflow.
The repository does not automatically run CI or models. The template shows
Ubuntu with Python 3.10/3.12, no dependency installation, `tests_public` only,
and the saved-result validator. Repository maintainers may enable it later with
appropriate GitHub permissions; historical model/test suites are excluded.
