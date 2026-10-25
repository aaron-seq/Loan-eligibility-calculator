# Maintainer index

LoanWise's existing [README](../README.md), [architecture](ARCHITECTURE.md) and
[roadmap](ROADMAP.md) describe the product. The assessment/request-boundary
slice (#39/#41/#42) merged through PR #44. Current follow-up covers source
evidence (#45) and the proposed reporting/scanning setup (#43); no new PRD
or stakeholder approval is implied.

| Purpose | Source of truth |
|---|---|
| Current follow-up and exit criteria | [Milestone 2](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/2), open #43/#45 |
| Completed assessment/request-boundary outcome | [Closed milestone 1](https://github.com/aaron-seq/Loan-eligibility-calculator/milestone/1) |
| Execution and sequencing | [Private Project 5](https://github.com/users/aaron-seq/projects/5), owner `aaron-seq` |
| Findings and implementation | [Repository issues](https://github.com/aaron-seq/Loan-eligibility-calculator/issues), [merged PR #44](https://github.com/aaron-seq/Loan-eligibility-calculator/pull/44) |
| Threat model, test strategy, commands and current checkpoint | [Security/testing evidence](security-testing-evidence.md) |
| Interface and upgrade contract | [API](API.md), [deployment/recovery](DEPLOYMENT.md) |
| Safe reporting and coverage gaps | [SECURITY.md](../SECURITY.md), [#43](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/43) |
| Artifact provenance and historical notices | [Provenance appendix](artifact-provenance.md), [#45](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/45) |

Project ID: `PVT_kwHOBcQCAc4BoYRq`. Native fields retain their meanings:
Status tracks workflow; Priority uses P1 high, P2 normal and P3 low; Milestone
references repository outcomes; labels classify area/type. No assignees, due
dates or estimates are invented. Todo is the combined triage/ready queue;
In Progress means active implementation; In review / Retest means a draft and
validation evidence exist; Done requires verified acceptance and justified
closure. Merge, release, deployment and exposure resolution are separate states.

Saved views combine current-milestone execution, triage/coverage gaps and
review/retest. Automatic issue closure and closed/merged-to-Done transitions
are absent; those decisions are manual. Existing add-item/sub-issue/link
workflows remain; do not infer delivery from their status changes. No timeline
or Insights chart is claimed: dates and velocity are unknown. Project privacy
does not make its public issue/PR items confidential.

Contributions use a dedicated branch, a linked issue, meaningful regressions
and the [README validation commands](../README.md#development-and-verification).
Use disposable data, inspect the complete diff, redact logs at collection and
follow SECURITY.md before recording sensitive details. Fresh agent verification
is useful evidence from the same system; it is not independent human approval.
Protected/default-branch merges, settings changes, release/deployment and
security disclosure require the appropriate explicit authorization.
