# LoanWise — explainable loan eligibility guidance

[![CI](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Code license](https://img.shields.io/badge/code-MIT-green)

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

Shared training/serving features, scikit-learn pipelines, repeated stratified CV
and training-only calibration selection support the estimates. Input ablation,
counterfactual guidance and separate affordability rules explain their limits.
Pydantic validation, CSRF, ownership checks and browser headers protect the
request boundary. See [architecture and contracts](docs/ARCHITECTURE.md).

The [model card](docs/MODEL.md) records metrics and sampling limits. Its holdout
results describe the evaluation estimator before the serving refit on all 614
rows; approval history does not measure repayment risk. Unsupported inputs return
`probability: null` and band `unavailable`, including saved history and comparison
rankings; rules remain available. See the [API contract](docs/API.md#out-of-range-applications)
and [schema upgrade](docs/DEPLOYMENT.md#schema-v2-upgrade).

## Model results and limits

Use the generated [model card](docs/MODEL.md) and [metrics](models/metrics.json)
as the authoritative results; historical benchmarks are in [BENCHMARKS.md](docs/BENCHMARKS.md).

## JSON API

With the server running, a cookie-free assessment needs no account or CSRF token:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/assess -H "Content-Type: application/json" --data-binary @examples/application.json
```

Use `curl.exe` in Windows PowerShell if `curl` resolves to a PowerShell alias.
The response includes probability, band, affordability, rules, factors,
suggestions, model provenance and a disclaimer. Signed-in requests also save
the result and require a CSRF token.

Verify the running server with `python scripts/smoke_http.py`.

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

Rate limits are per process; SQLite permits one writer at a time. Unfinished
capabilities and acceptance criteria are in the [roadmap](docs/ROADMAP.md).
The [maintainer index](docs/README.md) links verification evidence and delivery
tracking. Read [SECURITY.md](SECURITY.md) before reporting sensitive information.

## License

Project code: MIT — see [LICENSE](LICENSE) and preserved [historical notices](THIRD_PARTY_NOTICES.md).
Analytics Vidhya distributes matching CSVs; the original collector, historical
terms and dataset/model redistribution basis remain unresolved in
[#45](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/45); see the [provenance appendix](docs/artifact-provenance.md).
