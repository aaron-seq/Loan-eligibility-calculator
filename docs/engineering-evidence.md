# Repository review evidence — 2026-10-23

Base: [`7ea3a46`](https://github.com/aaron-seq/Loan-eligibility-calculator/commit/7ea3a460adb3b5e5da79f606308300fb30ffc24a).
Historical work branch: `codex/review-outputs-readme-issues`. The tests below
record its premerge verification. [PR #40](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/40)
subsequently passed all seven [CI jobs](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39922492056)
and merged with explicit authorization as [`fcdad2d`](https://github.com/aaron-seq/Loan-eligibility-calculator/commit/fcdad2dff99c587a49e4d17931afd29456f2afa5).
No deployment was performed. Current work is in the [security/testing checkpoint](security-testing-evidence.md#checkpoint-and-resume).

## Selected work and decisions

| Item | Evidence / root cause | Acceptance and benefit | Decision, risk and rollback |
|---|---|---|---|
| Integrate compatible PRs | Five open PRs; #35 supplies domain fixes and engineering gates; the others update actions or patch dependencies | Validate one combined branch with contributor history retained | Merge #35 locally; cherry-pick #37/#38 with attribution. #32/#33 already present through #35. No speculative ML library upgrade; revert integration commits to roll back |
| Prevent calibration selection on holdout | `train()` compared holdout Brier scores to choose calibration despite claiming no holdout selection | Select only from training rows; regression test supplies the opposite choice from holdout and verifies it is respected | Paired five-fold training CV with inner calibration folds. Extra training work, no per-request overhead. Revert trainer and regenerate artifacts together to roll back |
| Complete #36 | No response headers; inline percentage widths conflict with a strict CSP | Headers on HTML/JSON/static/redirect/error responses; working bars, styles and browser audit | One after-request hook and native progress values; no inline-style exemption. HSTS belongs at the TLS proxy. Revert hook/templates/CSS together to roll back |
| Make generated outputs portable | Windows model-card generation produced CP1252, failing UTF-8 decoding; checkout CRLF changed the dataset fingerprint | Generated model card reads as UTF-8; source fingerprint agrees with Linux | Explicit UTF-8/LF output and dataset LF checkout attributes; no CSV value changes |
| Verify startup and deployment | PR #35's Gunicorn test failed on Windows because `fcntl` is Unix-only; local Docker daemon unavailable | Portable entrypoint failure test plus Unix integration coverage and a real container gate | Windows explicitly skips Gunicorn; Linux CI retains it. Container job checks model readiness, custom-port health, headers, non-root execution and account persistence after container replacement |

Primary references consulted: [Flask response/session security](https://flask.palletsprojects.com/en/stable/web-security/),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html),
[model persistence/version requirements](https://scikit-learn.org/stable/model_persistence.html),
[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1),
[setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0),
[Pydantic 2.14.1](https://github.com/pydantic/pydantic/releases/tag/v2.14.1),
[Ruff 0.17.1](https://github.com/astral-sh/ruff/releases/tag/0.17.1),
[python-dotenv 1.2.4](https://github.com/theskumar/python-dotenv/releases/tag/v1.2.4).
The patches were accepted for the existing interfaces and verified together;
the model-library pins were retained.

## Output inventory

| Output | What was reviewed / demonstrated |
|---|---|
| [Model bundle](../models/loan_model.joblib), [metrics](../models/metrics.json), [generated card](MODEL.md) | Retrained on the pinned runtime. Logistic regression selected; CV AUC 0.728394 ± 0.053839, holdout AUC 0.8031, Brier 0.1702, accuracy 0.7967. Sampled monotonicity: 0/216 material reversals. Calibration selection training-CV Brier: 0.203740 uncalibrated vs 0.186747 calibrated |
| Dataset identity | `data/train.csv`: 614 rows, SHA-256 prefix `3ac62ced374efaf4`. LF-normalized working files are byte-identical to the base Git blobs; `.gitattributes` preserves that on future checkouts |
| Training semantics | Candidate and calibration selection use training rows only. The final serving estimator is refitted on all 614 rows; holdout numbers measure the evaluation estimator before refit. They are not an independent evaluation of that final refit |
| HTML and local CSS | Eleven rendered pages checked with Chromium/axe, including saved and admin pages; styles-loaded and no-CSP-blocked-resource assertions. Result page visually inspected at 1180px and 390px; no mobile horizontal overflow observed |
| [README input](../examples/application.json) through a live `python app.py` process | HTTP smoke on Windows returned model ready, probability 0.6734, band likely, instalment 14975.7, anonymous result not saved. Reusable check: [smoke_http.py](../scripts/smoke_http.py) |
| Comparison and unsupported inputs | Personally reproduced a zero-income `/api/assess` with `probability: 0.2` and `out_of_range: true`, and an all-zero-income comparison with `best_index: 0`. Tracked in [#39](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/39); no claim that this is resolved |
| Stored exploratory notebooks | [Decision tree](../notebooks/decision-tree-exploration.ipynb): 68 cells, 45 stored outputs, no stored error objects, stored accuracy 0.66 on 154 rows. [Exploration](../notebooks/exploratory-analysis.ipynb): 70 cells, 47 outputs, no stored error objects. Historical outputs inspected, **not reexecuted**; they do not evaluate the current serving model |
| [Benchmark report](BENCHMARKS.md) | New local 100-run measurement retained separately from #35's published before/after and HTTP throughput figures. No cross-machine speedup claim |

## Checks personally run

Windows 11, Python 3.13.7, a fresh isolated `.venv`, scikit-learn 1.9.1,
numpy 2.4.6. Test databases are disposable. Commands below assume the virtual
environment is active.

| Check | Command | Result |
|---|---|---|
| Base test suite | `python -m pytest --cov=loanapp --cov-report=term-missing` | 214 passed; 86.3% branch-aware coverage |
| Regression red/green | Targeted header, calibration and UTF-8 generation tests | Seen failing for missing headers, holdout selection and invalid UTF-8 respectively, then passing after fixes |
| Final combined suite | `A11Y_REQUIRED=1 python -m pytest --cov=loanapp --cov-report=term-missing -rs` (PowerShell: set `$env:A11Y_REQUIRED='1'` first) | **297 passed, 1 Unix-only Gunicorn skip; 93.0% branch-aware coverage**. All thirteen browser audit assertions passed, covering eleven pages |
| Types | `python -m mypy` | Clean, 18 source files, strict configuration |
| Lint / formatting | `python -m ruff check loanapp tests scripts app.py` and `python -m ruff format --check loanapp tests scripts app.py` | Clean |
| Static security | `python -m bandit -r loanapp app.py -ll` | No issues reported by this scan |
| Dependencies | `python -m pip_audit -r requirements.txt -r requirements-dev.txt --strict` | No known vulnerabilities reported for the resolved requirements on this date |
| Training | `python -m loanapp.ml.train` | Completed; regenerated bundle, metrics, UTF-8 card; results above |
| CSS | `TAILWINDCSS_VERSION=v3.4.19 python -m pytailwindcss -i static/src/tailwind.css -o <temporary-file> --content './templates/**/*.html' --minify` | Rebuilt; compared with committed output. The pinned binary reports an outdated Browserslist database warning; no template/CSP failure was observed |
| Live HTTP | `python scripts/smoke_http.py --base-url http://127.0.0.1:5017` | Passed against the development entrypoint and real sockets |
| Benchmark | `python scripts/benchmark.py --runs 100` | Three model calls/assessment; full-assessment p50 21.1ms, p95 43.4ms. Environment and limits in [BENCHMARKS.md](BENCHMARKS.md) |

Not personally run locally: Python 3.11/3.12, Unix Gunicorn serving, or Docker
build/startup (no running daemon). CI covers these separately; use the delivery
PR's actual check results, not this local table, as evidence of hosted runs.
No production deployment, external email/bureau integration, disaster recovery,
manual screen-reader audit or independent penetration test was performed.

## Every initially open issue

The ten original issues #1–#10 were already closed when this review began.
The fifteen open issues below were read, including discussion, and compared
with code/tests. Completed issues are linked for automatic closure on merge;
unfinished acceptance criteria stay visible.

| Issue | Disposition / remaining acceptance criteria |
|---|---|
| [#17 Password recovery / email verification](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/17) | Keep open: no outbound email, reset/verification tokens or recovery endpoints |
| [#18 Shared rate limits](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/18) | Keep open: counters remain process-local and reset with workers; a shared store and multi-worker tests are still needed |
| [#19 Comparison UI](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/19) | Keep open: API exists; no browser comparison view. Resolve #39 before presenting unsupported scenario rankings |
| [#20 Larger dataset](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/20) | Keep open: only 614 labelled rows; source/licence approval and new representative data are prerequisites |
| [#21 Repayment risk](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/21) | Keep open: no repayment labels; depends on #20. Approval prediction is not repayment risk |
| [#22 Accessibility](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/22) | Partly implemented: eleven-page automated browser audit and semantic/error fixes. Manual screen-reader acceptance criterion remains |
| [#23 Region/currency](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/23) | Keep open: no configurable currency labels or bureau-neutral API. README now states the assumptions |
| [#24 Postgres](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/24) | Keep open: SQLite only; no alternate backend, migration or database matrix |
| [#25 Shapley attribution](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/25) | Keep open: batched input ablation still ignores interactions; no SHAP implementation or measured trade-off |
| [#26 Drift monitoring](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/26) | Keep open: outcome aggregates are not input-distribution monitoring; reference distributions/windowed comparisons absent |
| [#27 Type checking](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/27) | Ready for closure on merge: strict mypy passes locally and is a CI gate; integrated from #35 |
| [#28 Export/share](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/28) | Keep open: no print/PDF export or revocable, expiring share link |
| [#29 Structured logging](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/29) | Keep open: text logging remains; request IDs/JSON formatter not implemented. Do not add applicant payloads to logs |
| [#30 Built CSS](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/30) | Ready for closure on merge: committed stylesheet, pinned standalone rebuild and freshness gate; integrated from #35 |
| [#36 Response headers](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/36) | Ready for closure on merge: strict CSP/frame/nosniff/referrer headers, native bars, HTML/JSON/error/static tests, browser regression checks |

New finding [#39](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/39)
remains open with reproduction, affected outputs, storage/migration implications
and acceptance tests. It was not silently folded into a documentation fix.

## Open PR review

| PR | Review / integration | Published CI inspected before integration |
|---|---|---|
| [#32 checkout v7](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/32) | Compatible; identical action update included through #35 | [Success](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/38538213606) |
| [#33 setup-python v7](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/33) | Compatible; identical action update included through #35 | [Success](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/38538226769) |
| [#35 Engineering fixes](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/35) | Reviewed code, tests, docs and discussion; merged locally with original commits. Added Windows startup portability, calibration-selection regression, output portability and #36. Prior benchmark/equivalence claims remain attributed to its author | [Success](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/38876001950) |
| [#37 Ruff patch](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/37) | Cherry-picked with Dependabot attribution; resolved requirements conflict by retaining #35's added tools plus Ruff 0.17.1; combined lint passes | [Success](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39236611876) |
| [#38 Web patches](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/38) | Cherry-picked with attribution; Pydantic 2.14.1 and python-dotenv 1.2.4 verified together | [Success](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39236627247) |

Closed PR #34's dotenv update is also present through #35. No open PR was
merged into `main` or closed as superseded before the combined delivery lands.
Individual green runs do not substitute for checks on the combined branch.
