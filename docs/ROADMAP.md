# Roadmap

Work that would genuinely improve the project, in the order I would do it.
Everything here is deliberately *not* done, with the reason stated — this is a
list of known gaps, not a wish list.

Each item is tracked as a GitHub issue; the number links to it.

## Known limitations shipping today

These are real and documented in code. None is hidden.

| Limitation | Impact | Where it is stated |
|---|---|---|
| Rate-limit counters are per process | N workers permit ~N× the limit | [#18](../../issues/18) |
| 614 training rows | Wide error bars on every metric | [#20](../../issues/20) |
| Attribution ignores interactions | Factors do not sum to the total | [#25](../../issues/25) |
| SQLite single-writer | Blocks horizontal scale | [#24](../../issues/24) |
| Tailwind from CDN | Unstyled offline | [#30](../../issues/30) |
| Exact dependency pins | Upgrades are manual | `requirements.txt`; weekly `pip-audit` in CI |
| No Content-Security-Policy | Inline scripts in templates | [#30](../../issues/30) |
| No type checking | A class of error the tests miss | [#27](../../issues/27) |
| No password reset | A locked-out user needs an operator | [#17](../../issues/17) |
| No email verification | Addresses are unverified | [#17](../../issues/17) |

---

## Next (highest value per unit of work)

### 1. Password reset and email verification ([#17](../../issues/17))

The most obvious gap in the account system. Registration accepts any address
without verifying it, and there is no way back in after a forgotten password.

Needs an outbound email path (SMTP or a provider), single-use expiring tokens,
and rate limiting on the reset endpoint. Deliberately deferred: it is the first
thing that requires infrastructure beyond a container, and every other feature
was reachable without it.

### 2. Shared-store rate limiting ([#18](../../issues/18))

Move `RateLimiter` behind Redis. The interface in `security.py` is already the
one a Redis backend would implement, so this is a backend swap plus a config
option, not a redesign.

Worth doing as soon as a deployment runs more than one worker in an environment
that attracts real traffic.

### 3. A scenario-comparison UI ([#19](../../issues/19))

`POST /api/compare` exists, is tested, and has no interface. A side-by-side view
— "what if I borrow less versus stretch the term" — is the most useful thing
still missing from the applicant experience, and the backend is already done.

### 4. Postgres support ([#24](../../issues/24))

See `docs/DEPLOYMENT.md` for the migration triggers. Contained to
`loanapp/db.py`; the domain and HTTP layers do not touch SQL.

---

## Model quality

### 5. Shapley values for attribution ([#25](../../issues/25))

The current one-at-a-time ablation is exact, cheap and honest, but it misses
interactions. `shap.KernelExplainer` over the calibrated pipeline would capture
them.

Deferred because it adds a substantial dependency, needs a background-sample
cache, and increases per-request latency by roughly an order of magnitude for
what is a refinement rather than a fix. The current method's limits are stated
plainly rather than obscured, which is the acceptable interim.

### 6. A better dataset ([#20](../../issues/20))

This is the single largest constraint on the whole project. 614 rows with a
0.73 cross-validated ROC-AUC and ±0.054 fold-to-fold spread cannot support
strong claims, and no amount of modelling work changes that.

Options, roughly in order of value: Lending Club's public issuance data
(~2M rows, real outcomes, includes interest rates and default labels); Fannie
Mae's single-family loan performance data; or a synthetic generator calibrated
to published aggregate statistics.

A larger dataset would also make several currently-deferred things worthwhile:
gradient boosting, isotonic instead of sigmoid calibration, per-segment
calibration, and a genuine default-risk model rather than an approval-mimicking
one.

### 7. Predict risk, not approvals ([#21](../../issues/21))

The model currently learns *"would a lender have approved this?"* — which
reproduces historical lender behaviour, including its biases. The more useful
and more defensible target is *"would this loan be repaid?"*.

That needs outcome data (item 6). It would let the tool tell an applicant
something a lender's approval pattern cannot: whether the loan is a good idea.

### 8. Drift monitoring ([#26](../../issues/26))

`/admin/stats` shows the outcome mix, which surfaces gross shifts. Proper
monitoring would compare the live input distribution against the training
distribution per feature and alert on divergence.

Cheap to add once there is enough traffic for the signal to be meaningful.

---

## Product

### 9. Export an assessment ([#28](../../issues/28))

PDF or a shareable link, so an applicant can take the result to a conversation
with a lender or a family member.

### 10. Localisation and currency ([#23](../../issues/23))

Currency is unlabelled throughout, and the credit-score bands are CIBIL's
(India). Both are hard-coded assumptions. Making region a configuration
concern — currency symbol, bureau score range, typical interest rates,
affordability conventions — would make the repository genuinely reusable as a
template for similar tools elsewhere, which is one of its stated goals.

### 11. Accessibility audit ([#22](../../issues/22))

The templates use semantic HTML, labelled inputs and reasonable contrast, but no
screen-reader testing has been done and there is no automated axe check in CI. A
tool that gives people financial guidance should not exclude anyone; this
deserves to be higher than its position here reflects.

---

## Engineering

### 12. Built CSS instead of the CDN ([#30](../../issues/30))

Removes a network dependency and a third-party script tag, and makes the page
work offline. Costs a Node build step, which the project currently does not have
at all — that simplicity is worth something and is why this has not been done.

### 13. Type checking in CI ([#27](../../issues/27))

The codebase is annotated throughout but `mypy`/`pyright` does not run. Adding it
would catch a class of error the tests do not.

### 14. Structured logging ([#29](../../issues/29))

Logs are human-readable strings. JSON lines with request IDs would make them
useful to an aggregator. Worth doing at the point where anyone is actually
aggregating them.

---

## Explicitly not planned

| Idea | Why not |
|---|---|
| Real bureau API integration | Regulated data, contractual access, and it would turn a demo into something people mistake for an underwriting system |
| Document upload / OCR | Large surface area, PII storage obligations, no clear benefit to the estimate |
| Microservices | Two thousand lines of Python. No. |
| Deep learning | 614 rows. A linear model is the right complexity, and the monotonicity screen would reject most alternatives anyway |
| Mobile app | The web UI is responsive. A native app would duplicate it for no user gain |
