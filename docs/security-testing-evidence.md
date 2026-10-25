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
| [#43](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/43) | Reporting/scanning configuration gap; Informational, P2 | External private intake and hosted scanning remain disabled/unconfigured in the latest settings read (2026-10-24). Four-feature setup proposal awaits authorization; see the issue and SECURITY.md |
| [#45](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/45) | Artifact provenance evidence gap; P2 | Matching Analytics Vidhya downloads verified on 2026-10-25. Original collector, historical terms and redistribution basis remain unverified; see the [appendix](artifact-provenance.md). Historical MIT notice preserved; no relicense or legal approval |

The [milestone](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/1)
requires #39/#41/#42 behavior, migration and trust-boundary regressions, real
browser checks, current-head CI and accurate consumer/upgrade documentation.
These criteria were met and the milestone closed after PR #44 merged and
post-merge CI passed. No production deployment or exposure-resolution claim
follows from that merge.

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

## Fresh verification and stability — 2026-10-24

Two fresh same-system agents reviewed public correctness/tests and
documentation/provenance. No independent human or legal
approval is implied. The public verifier reported no actionable defect after
40 focused tests and additional controlled JSON/range/cache/migration probes.
The orchestrator personally reran the public selection plus training tests:
**64 passed**, exit 0, then reproduced six reported migration rollback cases
(invalid JSON shapes, an invalid supported probability and orphan ownership).
An initial probability probe used an unsupported row that is intentionally
normalized; the corrected supported-row probe verified rollback as intended.

Personally run stability: **10 rounds × 39 passed**, one slow case deselected
per round, 165.7 seconds overall, with test-file order rotated. Cross-process
and browser tests remain separately exercised; these repeats are not a claim
of randomized test order. No flakes were observed in this bounded run. The
agent's larger exploratory totals are reported evidence, not substituted for
the personally rerun gating tests. Restricted results stay in restricted records.

The documentation agent identified the provenance/attribution gap. The
orchestrator verified current docs, the original Git notice and normalized
identities of both CSVs, created #45, preserved the full historical notice and
updated the generator/model card without changing data, metrics or model bytes.
At that stage no distributor or dataset rights were established. The later
matching-download evidence is in the [provenance appendix](artifact-provenance.md);
rights questions remain unresolved. A concise
[maintainer index](README.md) records the tracking and contribution workflow.

## Checkpoint and resume

Last verified: **2026-10-25**. [PR #44](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/44)
merged with explicit authorization on 2026-10-24 as main
[`bbf1441`](https://github.com/aaron-seq/Loan-eligibility-calculator/commit/bbf1441b9655626b612526d790219727642c5238).
All seven [final-head CI jobs](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/40080300294)
and all seven [post-merge CI jobs](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/40082400881)
passed. Main's tree matches the tested PR head `76f636c`. Issues #39/#41/#42
are closed as completed and milestone 1 is closed. No release, deployment or
advisory publication occurred.

Current follow-up branch: `codex/45-source-verification`, based on `bbf1441`.
It records primary-source download comparisons, updates the generated model-card
wording and reconciles stale delivery references. Runtime behavior, data, model,
metrics, dependency pins and license bytes are unchanged. Its delivery PR records
the final head, focused checks and corresponding CI; earlier runs do not validate
new commits.

Personally run for this follow-up: **24 training tests passed**, Ruff lint/format
passed and strict mypy passed across 18 source files. The model card was
regenerated from the existing metrics/bundle, without retraining. A fresh
read-only same-system agent independently reproduced the three download hashes,
row/field comparisons and card output, checked relative link targets, and found
no actionable diff issue. The orchestrator reproduced the download comparisons
and generation and verified that protected artifact bytes have no diff. This
review is technical evidence, not independent human review or rights approval.

The [private Project](https://github.com/users/aaron-seq/projects/5) has
#39/#41/#42 and PR #44 in **Done**. #43 remains Todo/P2 pending the owner's
settings decision; #45 remains open/P2 and moves to review when the source-evidence
draft is published. Source identity is partial progress, not redistribution
approval or acceptance of the issue's remaining criteria.
[Milestone 2](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/2)
holds this follow-up; it remains open while acceptance and owner decisions remain.
Saved views are a current-milestone board plus triage and review tables.
Description/README and statuses were read back. Automatic issue-close and
closed/merged-to-Done workflows remain absent; explicit acceptance/retest governs
Done. No owners or dates were invented. The first status-view filter update had
invalid GraphQL quoting; the corrected variable-based mutation was read back.
No chart, timeline or behavior beyond these checks is claimed.

Earlier review agents ran read-only; their material public test, rollback and
provenance claims were personally checked as above. The follow-up source review
and local platform diagnostic are recorded on the delivery PR when verified.
Confidential review/verification stays in restricted records. Current Git/PR
state overrides an older snapshot. Existing deployment, manual accessibility,
platform and scanner gaps remain stated above and in #43/#45.

Resume here: verify Git/PR/Project state and current-head CI for the follow-up
draft. Rerun appropriate checks after changes. Review nullable API consumers and
a pre-upgrade backup before any separately authorized rollout. #43 needs an
approved reporting/scanning setup; #45 needs source/terms evidence and a real
maintainer disposition. Continue the existing roadmap without closing unfinished
enhancements. Read restricted advisory/patch state only through authorized
GitHub access; do not copy it into public tracking.
