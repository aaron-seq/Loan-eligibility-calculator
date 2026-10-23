# Security and testing evidence — 2026-10-23

Bounded scope: assessment integrity, JSON request boundaries, account-data access,
browser response policy and SQLite upgrade/recovery. Base: main
[`fcdad2d`](https://github.com/aaron-seq/Loan-eligibility-calculator/commit/fcdad2dff99c587a49e4d17931afd29456f2afa5).
This is a publication-safe appendix, not a complete security assessment.

## Findings and delivery target

| Record | Classification / priority | Demonstrated behavior and acceptance |
|---|---|---|
| [#39](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/39) | Confirmed correctness defect; P1 | Unsupported scores appeared in API/history/comparison and aggregates. Return null/unavailable consistently, retain rules/costs, rank supported scenarios only, preserve supported historical results |
| [#41](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/41) | Confirmed input defect; Low technical impact, P2 | Non-object JSON raised request exceptions. Return controlled 4xx without writes; preserve valid JSON/forms and CSRF |
| [#42](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/42) | Privacy hardening / verification gap; P2 | Dynamic responses lacked `no-store`. Verify headers on private/account/error paths while retaining static caching. No cache disclosure was demonstrated |
| [#43](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/43) | Reporting/scanning configuration gap; Informational, P2 | External private intake is disabled; hosted scanning coverage is unavailable. Deferred configuration decision; see SECURITY.md |

The [milestone](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/1)
requires #39/#41/#42 behavior, migration and trust-boundary regressions, real
browser checks, current-head CI and accurate consumer/upgrade documentation.
No production deployment or exposure-resolution claim follows from a draft PR.

## Threat model and boundaries

Assets: account identities/password hashes, saved financial scenarios and their
ownership, model/data integrity, administrator portfolio data and CI credentials.
Anonymous users can assess/compare; registered users save/read/delete their own
scenarios; administrators additionally see the portfolio. CLI administrators and
operators control configuration, database backups and trusted model artifacts.

Browser/API input crosses schema and CSRF checks into the shared assessor and
parameterized SQLite repositories. Signed cookies carry identity; user state is
loaded from the database. Jinja renders stored labels and assessments. Training
uses the committed CSV offline; serving loads an operator-trusted joblib bundle.
There is no user-supplied model upload, remote retrieval, payment/webhook, LLM or
multi-tenant organization interface in the reviewed application.

| Threat / boundary | Verification and oracle | Disposition |
|---|---|---|
| User → another user's saved record; user → administrator | Real registration/login for two users and a CLI-created admin; owner-scoped reads/deletes and no forbidden changes | Selected paths verified in `test_trust_boundaries.py`; not all possible authorization cases |
| JSON → schema / CSRF | Representative array, string, numeric and boolean roots; valid-CSRF and missing-CSRF cases; no records/accounts created | #41 regressions pass |
| Stored label → HTML / SQL | Benign markup and apostrophe round-trip; escaped HTML; SQL parameterization reviewed | Selected stored-output case verified; not a general injection proof |
| Browser cache → private data | Dynamic `no-store`, static response unchanged | #42 header behavior verified; previously cached copies and actual proxy policy untested |
| Unsupported features → model guidance | Business invariant: the model must not report/rank an estimate it flags unsupported | #39 API, browser, persistence and aggregate regressions |
| Concurrent workers → SQLite migration | Frozen shipped v1 schema, actual SQLite files/processes, transaction failure and restored backup | Selected migration invariants verified; production size/disk/time unmeasured |
| Contributor → CI; operator → model | Read-only workflow permissions, ordinary `pull_request` trigger reviewed; model loading requires trusted artifacts | Configuration reviewed; compromised runner/artifact not simulated |

Actor × action × resource matrix, personally exercised with controlled fixtures:

| Action on owner's application | Owner | Other user | Admin account | Anonymous |
|---|---|---|---|---|
| API read / saved HTML | 200 / 200 | 404 / 404 | 404 / 404 | 401 / redirect |
| Prefill another owner's form | Own prefill covered by existing page tests | No owner amount disclosed | No owner amount disclosed | Sign-in required |
| Delete | Missing CSRF: 400, unchanged; valid CSRF: 200, then 404 | Valid CSRF: 404, owner's record unchanged | Not selected | Not selected |
| Admin statistics | Ordinary account: 403 | 403 | 200; portfolio HTML 200 | 401 |

Registration over-posting cannot set `is_admin`; disabling a controlled account
removes access on its next authenticated request. These are verified existing
protections, not newly discovered bypasses or new fixes.

## Standards and test map

References use [ASVS **5.0.0** requirements](https://raw.githubusercontent.com/OWASP/ASVS/v5.0.0/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.csv),
[WSTG **4.2**](https://wstg.owasp.org/v4.2/), and the
[API1:2023 object-authorization scenario](https://api-security.owasp.org/editions/2023/en/0xa1-broken-object-level-authorization/).
Selected requirements guide verification; this is not ASVS certification.

| Requirement / method | Test evidence | Oracle / limits |
|---|---|---|
| V2.2.1, V2.2.2 input validation | [Request boundaries](../tests/test_request_boundaries.py), existing schema/API tests | Invalid shapes produce 4xx, no writes; valid examples still work |
| V8.2.1, V8.2.2, V8.3.1 server-side authorization | [Actor tests](../tests/test_trust_boundaries.py) | Allowed identity confirmed through real login; deny object/function access and preserve owner's record |
| V1.2.1 output encoding; V1.2.4 query parameterization | Actor tests plus [repository review](../loanapp/db.py) | One stored label escaped/round-trips; no broad payload campaign |
| V3.5.1 CSRF | Actor/request tests and existing [security tests](../tests/test_security.py) | Missing token denies mutation; valid token allows owner action |
| V14.3.2 browser cache | Request boundary tests | `no-store` present on selected dynamic responses; not cache erasure |
| V2.3.3 transactional integrity | [Migrations](../tests/test_migrations.py), [concurrency](../tests/test_concurrency.py) | Failure leaves old records/schema intact; four workers upgrade once; backup restores v1 fixture |
| Model-supported output contract | [Assessment integrity](../tests/test_assessment_integrity.py) | No unsupported probability/band/ranking; empty/mixed portfolio handling |
| Actual rendered UI and CSP | [Chromium/axe](../tests/test_a11y.py) | 13 page states, real stylesheet guard and console CSP check; manual assistive-technology review remains [#22](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/22) |

Authentication/session lifecycle is **partially verified**, not passed as a
domain. Restricted security work, when present, belongs in the confidential
advisory workflow and is excluded from this public evidence. Password recovery
and email verification remain [#17](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/17);
shared rate limiting remains [#18](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/18).
Production TLS/proxy/cookie behavior, storage encryption, retention, real-host
recovery, load capacity, exhaustive injection testing and manual screen-reader
testing were **not tested**. OAuth/MFA/webhooks/uploads are not implemented;
their absence is not evidence that unrelated controls passed.

## Personally run checks and provenance

Local environment: Windows, Python 3.13.7, SQLite 3.50.4; pinned development
environment, disposable SQLite files and synthetic accounts only. No production
targets, real-user data or intrusive external tests. Model and dependency pins
are unchanged; no feature/training change requires a new model artifact.

Baseline main: **297 passed, 1 Unix-only Gunicorn skip**, 93.0% branch-aware
coverage. New defect tests first failed for the intended output, JSON and
header causes (16 failed/1 passed); four initial migration checks also failed
before implementation. Red runs were local and were not pushed as broken commits.
Final local result: **332 passed, 1 Unix-only Gunicorn skip**, 93.8% branch-aware
coverage, exit 0, no warnings. This includes the expanded backup/restore and
empty/mixed portfolio cases. Fixture-only SQLite connection warnings from an
earlier run were corrected with explicit connection closure. Hosted results
are recorded in the delivery PR.

Reproduction, from a pinned virtual environment at the PR head:

```powershell
$env:A11Y_REQUIRED='1'
python -m pytest --cov=loanapp --cov-report=term-missing -rs
python -m ruff check loanapp tests scripts app.py
python -m ruff format --check loanapp tests scripts app.py
python -m mypy
python -m bandit -r loanapp app.py -ll
python -m pip_audit -r requirements.txt -r requirements-dev.txt --strict
```

Tools: pytest 9.1.1, coverage 7.17.0, Ruff 0.17.1, mypy 2.4.0, Bandit 1.9.4,
pip-audit 2.10.1, Playwright 1.64.0, axe-playwright-python 0.1.8; Chromium v1248.
Ruff/type/static checks passed locally. Bandit scanned 3,307 lines with no
reported findings; its one existing B108 suppression is the documented
serverless `/tmp` constant in `config.py`. This pass added no suppressions.
Dependency audit results apply to both pinned requirement files and their
resolution at run time, not an assurance against unknown vulnerabilities.
The local audit completed with no known vulnerabilities reported. Real HTTP
smoke tests passed for the documented supported example (0.6734, likely,
instalment 14975.7), unsupported assessment, all-unavailable comparison and
`no-store`. A fresh pinned Tailwind v3.4.19 build matched the committed CSS;
the CLI emitted its upstream outdated Browserslist warning.

A bounded local credential-pattern check scanned the working tree and latest
50 HEAD commits: 307 unique text blobs up to 1 MiB, looking only for private-key
headers, GitHub token formats and AWS access-key IDs. No matches were detected.
It excluded binary/large blobs, artifacts, external logs and older history;
generic secrets and credentials outside those patterns remain unverified.
No values were printed or validated against external services.

Installed metadata was inspected for the principal runtime and test packages.
It identifies BSD/MIT/Apache and NumPy's bundled-license expression; several
packages lack an SPDX expression in metadata. This is an inventory check, not a
complete transitive attribution or legal review; no dependencies were added.
Hosted Dependabot/secret/code-scanning responses are recorded in #43 as gaps,
not clean scanner results. Local Docker has no running daemon; Linux container,
Unix startup and Python 3.11/3.12 verification use hosted CI. The personally
inspected [run for 074eb2a](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/39947629236)
passed all seven jobs. Its Python 3.13 job reports 318 passed/15 browser skips;
the dedicated browser job reports 15 passed (13 page states plus stylesheet/CSP
guards). The container job built the image and verified non-root execution,
custom-port health and account persistence after replacement. These published
results apply to that head; later assertion/documentation changes require their
own CI run, recorded on the PR.

## Checkpoint and resume

Checkpoint: 2026-10-23 17:20 Asia/Kolkata (11:50 UTC). Branch
`codex/39-assessment-integrity`, based on `fcdad2d`;
[draft PR #44](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/44)
contains the implementation and records the current head/check results.
Follow-up assertions explicitly check no-store on saved HTML, redirects and
error responses. Their targeted tests pass. No merge, release, advisory
publication or deployment is included in this bounded delivery authorization.

The [private Project](https://github.com/users/aaron-seq/projects/5) is linked to
the repository. #39/#41/#42 are In Progress for the milestone; #43 is Todo/P2.
Saved views: Current milestone, Triage and coverage gaps, Awaiting review.
Automatic issue closure and closed/merged-to-Done workflows were removed;
explicit acceptance/retest governs Done. No owners or dates were invented.
GitHub initially rejected one view mutation because its response field was
incorrect; the corrected operation was read back successfully. No duplicate
view was created by that failed validation.

Resume here: verify Git/PR/Project state, inspect current-head CI, and address
failures before requesting a merge decision. Run the commands above after any
code change; migration entry points are `tests/test_migrations.py` and
`loanapp/db.py`. Review nullable API consumers and take a pre-upgrade backup
before any separately authorized rollout. Continue #43 and the existing roadmap
after the bounded milestone; do not close unfinished enhancement issues.
Confidential advisory and remediation state must be read through authorized
GitHub access, not copied into this public checkpoint.
