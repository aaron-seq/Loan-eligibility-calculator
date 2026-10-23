# LoanWise — explainable loan eligibility guidance

[![CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Flask application for exploring historical loan approval patterns, monthly
repayments, and changes to an application. Applicants can assess anonymously;
an optional account saves results and lets them adjust a previous scenario.
Engineers can use the project to study a tested ML application with shared
training and serving contracts.

**Educational demonstration only.** The dataset contains 614 historical
applications and approval labels, not repayment outcomes. Estimates are neither
loan offers nor validated underwriting decisions. Gender and marital status are
excluded from the model; this does not establish fairness or legal compliance.
An optional CIBIL score affects published rules only, never the model.

## Quick start

Requires **Python 3.11–3.13**. Use the pinned dependencies: the committed model
bundle is checked against its training libraries at startup.

```bash
git clone https://github.com/aaron-seq/Loan-eligibility-calculator.git
cd Loan-eligibility-calculator
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
python app.py
```

On macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python app.py
```

Open [localhost:5000](http://127.0.0.1:5000), choose **Check eligibility**, and
submit the form. Income is monthly; the loan amount is total principal in the
same currency units, not thousands. The default rules use CIBIL-style bands
and a 10.5% reference interest rate; see [regional limits](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/23).

The bundle and stylesheet are committed, so first use needs neither training
nor a CSS build. Configuration comes from [.env.example](.env.example); real
environment variables take precedence. Administrators are created explicitly:

```bash
python -m flask --app app create-admin --user-id yourname --email you@example.com
```

The CLI prompts for a password. There is no default administrator account.

## Implemented techniques

| Technique | What it provides | Implementation / evidence |
|---|---|---|
| Shared feature engineering and versioned model bundles | The same eleven features, order, training rate and library checks at serving time | [Features](loanapp/features.py), [registry](loanapp/ml/registry.py), [architecture](docs/ARCHITECTURE.md) |
| Scikit-learn pipelines | Imputation and scaling fitted inside each validation fold | [Trainer](loanapp/ml/train.py) |
| Repeated stratified cross-validation | Candidate comparison with fold variability, plus a separate 20% holdout | [Generated model card](docs/MODEL.md) |
| Sigmoid calibration selected on training CV | Paired outer-fold Brier comparison; inner folds fit calibration; holdout labels do not select it | [Trainer tests](tests/test_training.py), [metrics](models/metrics.json) |
| Response-surface and fairness diagnostics | Loan-amount monotonicity screens and group outcome summaries; neither proves global monotonicity or fairness | [Model card](docs/MODEL.md) |
| Input ablation and counterfactual guidance | Batched rescoring explains inputs and ranks candidate changes; contributions ignore interactions and do not sum to the estimate | [Explainer](loanapp/explain.py), [benchmarks](docs/BENCHMARKS.md) |
| Separate affordability rules | Amortised instalments, interest and credit checks remain distinct from the model estimate | [Rules](loanapp/rules.py) |
| Validated requests and protected sessions | Pydantic schemas, scrypt passwords, CSRF, session rotation, SQL ownership checks, and per-process rate limits | [Schemas](loanapp/schemas.py), [security](loanapp/security.py) |
| Browser hardening and local styles | Strict CSP, frame protection, nosniff and referrer policy; committed Tailwind CSS without a runtime CDN | [Header tests](tests/test_response_headers.py), [stylesheet](static/src/tailwind.css) |
| Reproducible engineering checks | Strict mypy, Ruff, pytest/coverage, Chromium/axe, dependency audits, static scans, retraining and CSS freshness gates | [CI](.github/workflows/ci.yml), [review evidence](docs/engineering-evidence.md) |
| Stateful deployment | Flask app factory, blueprints, SQLite WAL, persistent Docker volume, non-root image and configurable workers | [Deployment](docs/DEPLOYMENT.md) |

## Model results and limits

The [generated model card](docs/MODEL.md) and [metrics JSON](models/metrics.json)
record the measured results and dataset fingerprint. The current calibrated
logistic regression has holdout ROC-AUC **0.8031**, Brier loss **0.1702**, and
accuracy **0.7967** at threshold 0.65 on 123 cases. Training CV ROC-AUC is
**0.7284 ± 0.0538**. The served model passes the sampled loan-amount screen
with **0 material reversals in 216 comparisons**.

The final serving estimator is refitted on all 614 rows. The holdout metrics
describe the evaluation estimator before that refit. Small samples, historical
bias, and a lack of repayment data limit what these numbers establish.
Extreme inputs are flagged as outside the model's usable range; the browser
withholds their percentage. API consumers must inspect `out_of_range` before
using a score. See [API limitations](docs/API.md#out-of-range-applications).

## JSON API

With the server running, a cookie-free assessment needs no account or CSRF token:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/assess -H "Content-Type: application/json" --data-binary @examples/application.json
```

Use `curl.exe` in Windows PowerShell if `curl` resolves to a PowerShell alias.
The response includes probability, band, affordability, rules, factors,
suggestions, model provenance and a disclaimer. Signed-in requests also save
the result and require a CSRF token.

For that sample, the 2026-10-23 local HTTP run returned probability `0.6734`,
band `likely`, monthly instalment `14975.7`, and `saved: false`. Reproduce the
check with `python scripts/smoke_http.py` while the server runs.

`POST /api/compare` compares 2–5 scenarios through JSON. A comparison UI is
[planned](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/19).
Full endpoints, worked examples and contracts: [API guide](docs/API.md).

## Development and verification

Run these after activating the virtual environment; no Make installation is required:

```bash
python -m pytest --cov=loanapp --cov-report=term-missing -rs
python -m ruff check loanapp tests scripts app.py
python -m ruff format --check loanapp tests scripts app.py
python -m mypy
python -m bandit -r loanapp app.py -ll
python -m pip_audit -r requirements.txt -r requirements-dev.txt --strict
python -m playwright install chromium
python -m pytest -m a11y -rs
```

Set `A11Y_REQUIRED=1` to fail when the browser audit cannot run. Gunicorn's
startup integration test runs on Unix; Windows checks the Python entrypoint
and skips that Unix-specific test. Manual screen-reader review remains open
in [#22](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/22).

Retrain with `python -m loanapp.ml.train` after changing the feature contract or
ML libraries; commit the bundle, metrics and generated model card together.
After changing template classes, rebuild CSS with `make css` on a Make-capable
shell, or use the [direct Python command](docs/ARCHITECTURE.md#building-the-stylesheet).
Keep `TAILWINDCSS_VERSION=v3.4.19`. [Benchmarks](docs/BENCHMARKS.md) provide
reproduction commands and distinguish prior measurements from this review.

## Deployment and remaining work

Deploy one container with a writable persistent volume, a production secret,
and HTTPS at the reverse proxy. See [deployment and recovery](docs/DEPLOYMENT.md)
for Docker commands, admin creation, backups, proxy settings and health checks.
`GET /api/health` returns HTTP 200 even when the model is degraded; monitor its
`status` and `model_status` fields. Gunicorn production serving requires Unix
or a Linux container.

Rate limits remain per process and SQLite permits one writer at a time.
Password recovery, a shared rate-limit store, multi-instance database support,
larger outcome datasets and other unfinished capabilities are in the
[roadmap](docs/ROADMAP.md). The [review evidence](docs/engineering-evidence.md)
records issue/PR dispositions, tests personally run and remaining validation gaps.

## License

MIT — see [LICENSE](LICENSE). Original author: Aaron Emmanuel Xavier Sequeira.
