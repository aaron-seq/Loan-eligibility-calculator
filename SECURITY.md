# Security reporting and support

No release/backport support commitment is currently documented. The development
runtime is Python 3.11–3.13 with the repository's pinned dependencies; production
serving uses Unix/Gunicorn or the Linux container. An educational demo is not a
validated underwriting system.

As checked on 2026-10-23, GitHub private vulnerability reporting is **disabled**
for this repository. No external private contact has been verified. Do not post
credentials, personal information or sensitive reproduction details in public
issues or PRs. The reporting and scanning setup gap is tracked in
[#43](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/43).

Repository administrators can use GitHub's confidential **draft repository
security advisory** workflow; draft creation and restricted visibility were
verified. This does not provide external reporters with an intake channel.
Do not publish an advisory or request disclosure as part of ordinary triage.
See [GitHub's advisory guidance](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/about-repository-security-advisories).

Safe, ordinary defects and test gaps may use repository issues. CI runs
`pip-audit` and Bandit; their results apply to their scanned inputs and do not
establish absence of vulnerabilities. See the [review scope and evidence](docs/security-testing-evidence.md).
