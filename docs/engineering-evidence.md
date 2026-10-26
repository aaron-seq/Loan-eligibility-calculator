# Repository review evidence — 2026-10-23

Historical validation of [PR #40](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/40),
from base `7ea3a46` to merged `fcdad2d`. All seven [delivery CI jobs](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39922492056)
passed. No deployment occurred. The [complete original record](https://github.com/aaron-seq/Loan-eligibility-calculator/blob/de44c8bf89a30d98175b8e13e439b3690260add4/docs/engineering-evidence.md)
retains the dated inventory; current scope is in the [security/testing checkpoint](security-testing-evidence.md#checkpoint-and-resume).

## Selected work and decisions

- Integrated #35 with contributor history; cherry-picked #37/#38 with attribution. #32/#33 were already included through #35; model-library pins stayed unchanged.
- Selected calibration using paired training CV with inner calibration folds, keeping holdout labels out of selection. Regression tests first demonstrated the defect.
- Added strict browser headers and native progress bars without inline-style exemptions; #36 completed.
- Generated the model card as UTF-8/LF and normalized CSV fingerprints across checkouts without changing values.
- Retained Unix Gunicorn coverage and added portable entrypoint plus container persistence checks. Windows explicitly skips Unix-only serving.

Revert the relevant integration commits to undo these changes; model/code
contracts must roll back together. Current upgrade/data recovery is documented
in [deployment](DEPLOYMENT.md#schema-v2-upgrade).

## Output inventory

The [model card](MODEL.md) and [metrics](../models/metrics.json) are authoritative
for results. Candidate/calibration selection uses training rows only; holdout
metrics describe the evaluation estimator before the final all-row serving refit.
Dataset identities and rights gaps live in [provenance](artifact-provenance.md).

The live [sample request](../examples/application.json) returned probability
0.6734, band likely, instalment 14975.7 and `saved: false`. Eleven rendered page
states were audited; the result page was visually checked at 1180px/390px with
no observed mobile overflow. These observations do not establish all browser or
assistive-technology behavior.

Historical notebooks were inspected, not rerun: [decision tree](../notebooks/decision-tree-exploration.ipynb)
has 68 cells/45 outputs, stored accuracy 0.66 on 154 rows; [exploration](../notebooks/exploratory-analysis.ipynb)
has 70 cells/47 outputs. Neither had stored error objects; neither evaluates the
current serving model. Their unique exploratory outputs remain preserved.
[Benchmarks](BENCHMARKS.md) distinguish personal measurements from #35's claims.

## Checks personally run

Windows 11, Python 3.13.7, scikit-learn 1.9.1, NumPy 2.4.6; disposable databases.
Use the current [verification commands](../README.md#development-and-verification).

| Check | Dated result |
|---|---|
| Baseline | 214 passed; 86.3% branch-aware coverage |
| Regression red/green | Header, holdout-selection and UTF-8 tests failed for the intended causes, then passed |
| Combined suite with required browser | 297 passed, one Unix Gunicorn skip; 93.0% coverage; 13 browser assertions over 11 pages |
| Ruff / formatting / strict mypy | Passed; 18 typed source files |
| Bandit / pip-audit | No findings / known vulnerabilities reported for the scanned inputs on that date |
| Training / pinned CSS rebuild | Completed; CSS matched; upstream Browserslist warning remained |
| Real HTTP / benchmark | Smoke passed; 100 runs, three model calls/assessment, p50 21.1ms / p95 43.4ms |

Python 3.11/3.12, Unix Gunicorn and Docker were not personally run locally;
published CI is separate evidence. Production deployment, external email/bureau
integration, actual-host recovery, manual screen-reader review and independent
penetration testing were not performed.

## Every initially open issue

#27/#30/#36 completed through PR #40. The unsupported-output defect discovered
as [#39](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/39) later
completed through PR #44. Remaining feature requirements are maintained once in
the [roadmap](ROADMAP.md) and linked issues; automated accessibility did not close
the manual acceptance gap in #22. Historical dispositions are in the original
dated record linked above.

## Open PR review

| Original PR | Integration / published CI inspected |
|---|---|
| [#32](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/32) | Checkout v7 included through #35; [CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/38538213606) |
| [#33](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/33) | Setup-python v7 included through #35; [CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/38538226769) |
| [#35](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/35) | Engineering integration, original commits retained; [CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/38876001950) |
| [#37](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/37) | Ruff 0.17.1, attribution retained; [CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39236611876) |
| [#38](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/38) | Pydantic 2.14.1 / dotenv 1.2.4, attribution retained; [CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39236627247) |

Closed #34's dotenv update was included through #35. Individual CI runs were
not substituted for validation of the combined delivery.
