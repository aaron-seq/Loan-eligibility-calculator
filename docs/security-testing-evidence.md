# Security and testing evidence — 2026-10-23

Public evidence for assessment integrity, JSON boundaries, account-data access,
response caching and SQLite recovery. This is a bounded assessment, not a claim
of repository-wide security. The [original dated record](https://github.com/aaron-seq/Loan-eligibility-calculator/blob/de44c8bf89a30d98175b8e13e439b3690260add4/docs/security-testing-evidence.md)
retains intermediate runs; operational instructions live in [deployment](DEPLOYMENT.md).

## Findings and delivery target

| Record | Verified outcome / remaining criterion |
|---|---|
| [#39](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/39), P1 | Closed: unsupported estimates become null/unavailable across API, saved history, comparisons and aggregates; supported results remain intact |
| [#41](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/41), P2 | Closed: non-object JSON returns controlled 4xx with no writes; valid forms/JSON and CSRF retained |
| [#42](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/42), P2 | Closed: dynamic `no-store`, static caching retained; no prior cache disclosure was demonstrated |
| [#43](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/43), P2 | Open: private intake/hosted scanning setup proposal awaits approval; configuration reads are not clean scan results |
| [#45](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/45), P2 | Open: matching distributor/downloads verified; original collection, historical terms and redistribution evidence remain unresolved |

[Milestone 1](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/1)
closed after #39/#41/#42 acceptance, migration/browser regressions, documentation
and current-head CI. Merge is separate from release, deployment and exposure resolution.

## Threat model and boundaries

Assets: identities/password hashes, owned financial scenarios, model/data
integrity, administrator portfolio data and CI credentials. Browser input crosses
schema/CSRF checks into the assessor and parameterized repositories; Jinja renders
stored values. Operators control backups and trusted joblib artifacts. There is
no upload, payment/webhook, OAuth/MFA or LLM interface to assess.

The selected actor matrix used two registered users, an administrator created
through the CLI and anonymous clients with disposable data:

| Action on owner's record | Owner | Other user | Administrator | Anonymous |
|---|---|---|---|---|
| API read / saved HTML | 200 / 200 | 404 / 404 | 404 / 404 | 401 / redirect |
| Prefill | Own record covered | No amount disclosed | No amount disclosed | Login required |
| Delete | Missing CSRF: 400, unchanged; valid: 200 then 404 | Valid CSRF: 404, record unchanged | Not selected | Not selected |
| Admin statistics | 403 | 403 | 200, portfolio HTML 200 | 401 |

Over-posted registration cannot set administrator status; disabling an account
denies its next authenticated request. A benign stored markup/apostrophe label
round-trips and escapes in HTML. These are selected existing-control checks,
not general injection or authorization proofs.

## Standards and test map

References: [ASVS 5.0.0](https://raw.githubusercontent.com/OWASP/ASVS/v5.0.0/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.csv),
[WSTG 4.2](https://wstg.owasp.org/v4.2/) and [API1:2023](https://api-security.owasp.org/editions/2023/en/0xa1-broken-object-level-authorization/).
Selected requirements guide tests; no certification is claimed.

| Boundary / requirement | Test and oracle |
|---|---|
| Validation, V2.2.1/V2.2.2; CSRF, V3.5.1 | [Request tests](../tests/test_request_boundaries.py): representative JSON roots return 4xx/no writes; valid requests succeed |
| Authorization, V8.2.1/V8.2.2/V8.3.1 | [Actor tests](../tests/test_trust_boundaries.py): real login and owner-scoped read/delete; deny cross-account/admin actions |
| Encoding/SQL, V1.2.1/V1.2.4 | Actor fixture escapes/round-trips; parameterized [repository](../loanapp/db.py) reviewed |
| Cache, V14.3.2 | Request tests verify selected dynamic `no-store`, retaining static policy; not cache erasure |
| Transactions, V2.3.3 | [Migration](../tests/test_migrations.py)/[concurrency](../tests/test_concurrency.py): shipped v1 fixture, rollback, four workers, backup restore and future-version refusal |
| Supported-output contract | [Integrity tests](../tests/test_assessment_integrity.py): null estimates, supported-only rankings, empty/mixed aggregates |
| Rendered UI/CSP | [Chromium/axe](../tests/test_a11y.py): 13 page states and stylesheet/CSP guards; manual review remains #22 |

Session lifecycle is partially verified; confidential work belongs only in
restricted advisory records. Password/email recovery (#17) and shared rate
limiting (#18) remain open. Production TLS/proxy/cookies, encryption/retention,
actual-host recovery, capacity, exhaustive injection and manual screen-reader
testing were not exercised. Missing features are not passed controls.

## Personally run checks and provenance

Windows, Python 3.13.7, SQLite 3.50.4; pinned tools and disposable databases/accounts.
No production targets or real-user data. Commands are maintained in the
[README](../README.md#development-and-verification); browser-required runs set
`A11Y_REQUIRED=1`. Data/model/dependency pins were unchanged for this delivery.

| Check | Dated evidence |
|---|---|
| Baseline / final | 297 passed + one Unix skip, 93.0% coverage → 332 passed + one Unix skip, 93.8% branch-aware coverage |
| Red regressions | Initial output/JSON/header cases: 16 failed/one passed; four migration cases also failed before implementation |
| Ruff / format / strict mypy | Passed |
| Bandit / dependency audit | 3,307 scanned lines, no findings; no known vulnerabilities reported for both pinned requirements at run time |
| HTTP / CSS | Supported example, unsupported/mixed comparisons and `no-store` passed; pinned CSS rebuilt identically with upstream Browserslist warning |

Tools: pytest 9.1.1, coverage 7.17.0, Ruff 0.17.1, mypy 2.4.0, Bandit 1.9.4,
pip-audit 2.10.1, Playwright 1.64.0, axe-playwright-python 0.1.8, Chromium 1248.
The existing B108 suppression documents the serverless `/tmp` fallback; no new
suppression was added. Principal package metadata was inventoried, not a complete
transitive legal review.

A bounded credential-pattern check covered the working tree and 50 commits:
307 unique text blobs up to 1 MiB, private-key headers/GitHub tokens/AWS key IDs.
No matches were detected or externally validated. Older history, binaries,
large blobs, generic secrets, artifacts and external logs remain unverified;
no values were printed. Hosted coverage gaps remain #43.

## Fresh verification and stability — 2026-10-24

Same-system read-only agents reviewed correctness and provenance; this is not
human/legal approval. The public verifier reported 40 focused tests and exploratory
probes. The orchestrator reran 64 selected/training tests and six migration
rollback cases, correcting an initial probe that used an intentionally normalized
unsupported row. Ten rounds of 39 tests (one slow case deselected/round) passed
in 165.7 seconds, rotating file order; no observed flake, not randomized/exhaustive testing.

Primary-source download comparisons and generated-card reproducibility later
completed through PR #46. Hashes, notices and remaining rights questions have
one maintained location: [artifact provenance](artifact-provenance.md).

## Checkpoint and resume

Verified 2026-10-27: PR #44 merged as `bbf1441` with all seven [post-merge jobs](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/40082400881)
passed. PR #46 merged as `de44c8b`, also with seven [post-merge jobs](https://github.com/aaron-seq/Loan-eligibility-calculator/actions/runs/40223312823)
passed; its tree matches tested head `e79cca7`. Inspected published logs for that
head report 318 passed/15 browser skips plus 15 passed in the dedicated browser
job. Hosted checks cover Python 3.11–3.13, Unix startup and container
non-root/custom-port health/persistence; local Docker verification remains unavailable.

Current cleanup: [#50](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/50),
branch `codex/50-repository-cleanup`, base `de44c8b`. Its PR records measured
reductions and final validation; this historical evidence does not validate new
commits. [Milestone 2](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/2)
remains open for #43/#45. No release, deployment or advisory publication occurred.

Resume from live Git/PR/current-head CI, using the [maintainer index](README.md).
Apply [nullable API](API.md#out-of-range-applications) and [backup/upgrade](DEPLOYMENT.md#schema-v2-upgrade)
contracts before any separately authorized rollout. Keep confidential patch
details out of public tracking; retain unfinished roadmap acceptance criteria.
