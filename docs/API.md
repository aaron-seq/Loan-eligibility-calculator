# API reference

The web UI is a client of this API. Anything the browser can do, a script can.

## Conventions

**Success** — `200`/`201`, payload under `data`:

```json
{ "data": { … } }
```

**Failure** — a machine-readable `error` and a human-readable `message`:

```json
{ "error": "validation_failed", "message": "Check the highlighted fields.",
  "errors": { "loan_amount": "Input should be greater than 0" } }
```

| `error` | Status | Meaning |
|---|---|---|
| `validation_failed` | 400 | Input rejected; `errors` maps field → message |
| `csrf_failed` | 400 | Session cookie present without a valid token |
| `unauthorised` | 401 | Sign-in required |
| `forbidden` | 403 | Signed in, but not an administrator |
| `not_found` | 404 | No such endpoint or record |
| `rate_limited` | 429 | Includes a `Retry-After` header |
| `model_unavailable` | 503 | No usable model; `message` says what to do |
| `internal_error` | 500 | Logged server-side; never echoes exception text |

### Authentication

Session cookie, set by `POST /login` or `POST /register`.

### CSRF

Required only for requests that **carry a session cookie**. A cookie-less client
(plain `curl`, a server-side integration) needs no token — CSRF is an attack on
ambient credentials, and such a request has none.

Authenticated clients fetch a token from `GET /api/csrf-token` and send it as
`X-CSRF-Token`.

### Rate limits

Per client IP: login 10/5min · registration 5/hour · assessment 30/hour.
Counters are per process; see the limitation note in `docs/ARCHITECTURE.md`.

---

## `POST /api/assess`

Assess one application. **No account required.** If a session is present the
result is saved to that user's history.

### Request

| Field | Type | Required | Notes |
|---|---|---|---|
| `applicant_income` | number | yes | **Monthly**, currency units |
| `coapplicant_income` | number | no | Monthly. Default 0 |
| `loan_amount` | number | yes | **Total principal**, currency units — not thousands |
| `loan_term_months` | integer | yes | 6–480 |
| `credit_history` | integer | yes | `1` = no adverse record, `0` = adverse or none |
| `dependents` | integer | no | 0–10. Default 0 |
| `education` | string | no | `graduate` \| `not_graduate`. Default `graduate` |
| `self_employed` | boolean | no | Default false. Accepts `"on"`/`"yes"`/`"1"` from forms |
| `property_area` | string | no | `rural` \| `semiurban` \| `urban`. Default `urban` |
| `cibil_score` | integer | no | 300–900. **Rules layer only — never a model input** |
| `label` | string | no | Names the scenario in saved history |

Accepts JSON or form-encoded bodies.

Note the units. The training CSV records loan amounts in thousands; the API takes
plain currency and converts internally. The old form asked users to pre-divide by
1000, which is a reliable way to collect wrong data.

### Response

The response below is illustrative, abbreviated, and is not a captured
measurement. The checked-in [sample input](../examples/application.json) and
[review evidence](engineering-evidence.md) provide a reproduced example.

```json
{
  "data": {
    "probability": 0.6421,
    "band": "borderline",
    "band_label": "Borderline",

    "affordability": {
      "monthly_instalment": 14975.42,
      "total_income": 60000.0,
      "emi_to_income": 0.2496,
      "total_repayable": 3594100.8,
      "total_interest": 2094100.8,
      "max_affordable_loan": 3004867.4,
      "annual_interest_rate": 0.105
    },

    "rules": [
      { "code": "emi_to_income", "severity": "pass",
        "title": "Instalment is affordable",
        "detail": "The instalment is 25% of household income, within the 50% ceiling most lenders apply." }
    ],
    "rules_verdict": "warn",

    "factors": [
      { "feature": "loan_amount", "label": "The amount you want to borrow",
        "impact": -0.1832, "direction": "hurts",
        "detail": "At 1,500,000 this is higher than a typical application…" }
    ],

    "suggestions": [
      { "action": "Borrow 10% less",
        "detail": "Requesting 1,350,000 instead of 1,500,000.",
        "projected_probability": 0.6612, "delta": 0.0191 }
    ],

    "model": {
      "version": "logistic_regression-20260729-3ac62ced374efaf4",
      "calibrated": true,
      "metrics": { "roc_auc": 0.8031, "brier_score": 0.1702, "accuracy": 0.7967, "holdout_rows": 123 }
    },

    "assessed_at": "2026-07-29T06:30:00+00:00",
    "saved": false,
    "save_hint": "Sign in to keep this assessment in your history.",
    "disclaimer": "This is an educational estimate…"
  }
}
```

Field notes:

- **`probability`** — calibrated P(approve), or `null` outside the model's usable range. `model.metrics` and
  `model.calibrated` describe how much weight it deserves.
- **`band`** — `likely` (≥ 0.65) · `borderline` (≥ 0.40) · `unlikely`, or `unavailable` when no estimate is supported. Thresholds
  are configurable.
- **`rules`** — `severity` is `pass` \| `warn` \| `fail`. Independent of the
  model; `rules_verdict` is the worst severity present. A `fail` here does not
  override the probability — both are reported.
- **`factors`** — ranked by absolute impact, capped at 5. `impact` is a signed
  probability delta from setting that one input to its training-set baseline.
  **Attributions do not sum to `probability`** — they are measured one at a time
  and ignore interactions.
- **`suggestions`** — empty when the applicant is already in the `likely` band.
  Each `projected_probability` is a real re-score, not an estimate. At most one
  "borrow less" option is offered: the smallest reduction that clears the band.
- **`application_id`** — present only when `saved` is true.

### Out-of-range applications

`POST /api/assess` includes `out_of_range` and `out_of_range_reason`.
An application with no household income, or an implausible loan/instalment
ratio, has no supported probability. Since application version **2.2.0**,
these results return `probability: null`, `band: "unavailable"`,
`band_label: "Estimate unavailable"`, and empty `factors`/`suggestions`.
Affordability, rules, reason and model provenance remain available.
This intentionally changes the previous numeric-score contract: consumers must
handle nullable probabilities and the additional band before upgrading.
[#39](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/39)
tracks the change.

The same invariant holds in saved history, including records already flagged
unsupported by the schema-v1 application. The schema-v2 upgrade preserves
supported history without rescoring it; see [upgrade and recovery](DEPLOYMENT.md#schema-v2-upgrade).

---

## `POST /api/compare`

Score 2–5 scenarios side by side.

```bash
curl -s localhost:5000/api/compare -H 'Content-Type: application/json' -d '{
  "scenarios": [
    {"applicant_income": 50000, "loan_amount": 2000000, "loan_term_months": 240, "credit_history": 1},
    {"applicant_income": 50000, "loan_amount": 1200000, "loan_term_months": 240, "credit_history": 1}
  ]
}'
```

```json
{ "data": {
    "scenarios": [
      { "index": 0, "probability": 0.4103, "band": "borderline",
        "monthly_instalment": 19967.2, "emi_to_income": 0.3993, "application": { … } }
    ],
    "best_index": 1,
    "disclaimer": "…"
} }
```

Validation errors are keyed by index: `scenarios.1.loan_amount`.
Each scenario includes `out_of_range` and `out_of_range_reason`; unsupported
scenarios have a null probability and unavailable band. `best_index` selects
only supported scenarios and is `null` when none qualify. Instalments remain
available even when the model cannot estimate an outcome.

---

## `GET /api/health`

No authentication. Returns **200 even when degraded** — alert on `status`, not
the HTTP code. See `docs/DEPLOYMENT.md`.

---

## `GET /api/model-card`

What is running and how well it performs: version, calibration, training
timestamp, holdout metrics, the exact feature list, and the disclaimer. Useful
for a status page or an audit trail. The full narrative model card is
`docs/MODEL.md`.

---

## `GET /api/csrf-token`

```json
{ "data": { "csrf_token": "…" } }
```

Only needed by clients that hold a session cookie.

---

## Applications (authentication required)

| Endpoint | Description |
|---|---|
| `GET /api/applications` | The caller's saved assessments, newest first (max 50) |
| `GET /api/applications/<id>` | One saved assessment, with the full stored result |
| `DELETE /api/applications/<id>` | Delete one |

Every query is scoped by `user_id` **in the SQL**. Another user's application
returns `404`, not `403` — existence is not disclosed.

Supported saved assessments are returned as recorded. They are not re-scored
against a newer model. The schema-v2 upgrade removes unsupported model claims
from historical records already flagged out of range.
`model_version` records which model produced it; estimates from different
versions are not directly comparable.

---

## Admin (administrator required)

`GET /admin/stats` — aggregate counters only, no per-applicant rows, so it is
safe to scrape:

```json
{ "data": {
    "total": 128, "scored": 128, "unavailable": 0, "applicants": 41, "mean_probability": 0.58,
    "bands": { "likely": 61, "borderline": 40, "unlikely": 27 },
    "by_day": [ { "day": "2026-07-29", "count": 12 } ],
    "model_versions": [ { "version": "…", "count": 128 } ],
    "users": 41, "model_status": "ready"
} }
```

Administrators are created with `flask --app app create-admin`. There is no
default account.
`mean_probability` and `bands` use supported results only; the mean is `null`
when `scored` is zero. Total, applicant, date and model-version counts still
include unavailable results.

---

## Worked example: sign in, assess, review

```bash
COOKIES=$(mktemp)

# 1. Register (needs a token — the client now holds a session cookie)
TOKEN=$(curl -s -c $COOKIES -b $COOKIES localhost:5000/api/csrf-token | python -c 'import json,sys;print(json.load(sys.stdin)["data"]["csrf_token"])')
curl -s -c $COOKIES -b $COOKIES localhost:5000/register \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"alice\",\"email\":\"alice@example.com\",\"full_name\":\"Alice\",
       \"password\":\"correct-horse-battery\",\"confirm_password\":\"correct-horse-battery\",
       \"csrf_token\":\"$TOKEN\"}"

# 2. Assess — saved to Alice's history
TOKEN=$(curl -s -c $COOKIES -b $COOKIES localhost:5000/api/csrf-token | python -c 'import json,sys;print(json.load(sys.stdin)["data"]["csrf_token"])')
curl -s -c $COOKIES -b $COOKIES localhost:5000/api/assess \
  -H "Content-Type: application/json" -H "X-CSRF-Token: $TOKEN" \
  -d '{"applicant_income":50000,"loan_amount":1500000,"loan_term_months":240,"credit_history":1}'

# 3. Review
curl -s -b $COOKIES localhost:5000/api/applications
```
