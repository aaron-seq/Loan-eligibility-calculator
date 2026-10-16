# Deployment

## The short version

**Deploy the container.** The application is stateful — accounts, saved
applications, an admin view — and a container gives it a real filesystem and a
persistent volume.

Serverless (Vercel, Lambda) is **not a supported target** and `vercel.json` has
been removed. That is a real constraint, not a configuration problem, and the
reasoning is below.

---

## Required configuration

| Variable | Required | Notes |
|---|---|---|
| `SECRET_KEY` | **yes, in production** | ≥32 chars. The app refuses to start without it. |
| `FLASK_ENV` | yes | `production` \| `development` \| `testing` |
| `DATABASE_PATH` | no | Defaults to `./loanwise.db`, or `/tmp/loanwise.db` on serverless |
| `MODEL_DIR` | no | Defaults to `./models` |
| `TRUSTED_PROXY` | only behind a proxy | Makes rate limiting trust `X-Forwarded-For` |
| `WEB_CONCURRENCY` | no | gunicorn worker processes (Docker default 2). **Set to the CPU count**: throughput scales linearly with processes, not threads — see [`BENCHMARKS.md`](BENCHMARKS.md) |
| `RATE_LIMIT_PREDICT` | no | `<requests>/<seconds>`, default `30/3600`. Raise it if many users share one IP (corporate NAT) |
| `RATE_LIMIT_LOGIN` / `RATE_LIMIT_REGISTER` | no | Same format; defaults `10/300` and `5/3600`. A malformed value stops the app from starting |
| `SESSION_COOKIE_SECURE` | no | Defaults to true in production |
| — | — | Requires Python >= 3.11 (set by numpy/pandas/scikit-learn) |
| `ANNUAL_INTEREST_RATE` | no | Fraction, e.g. `0.105`. Affordability reference rate. |
| `APPROVE_THRESHOLD` / `BORDERLINE_THRESHOLD` | no | Decision bands, `0 < borderline < approve < 1` |

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

**There is no fallback.** The previous version defaulted to `os.urandom(24)`,
which looks secure and silently breaks every session the moment a second worker
starts with a different key — an intermittent "randomly logged out" bug that is
miserable to diagnose. Failing at boot with a message is better.

---

## Docker (recommended)

```bash
docker build -t loanwise .

docker run -d --name loanwise -p 8000:8000 \
  -e WEB_CONCURRENCY="$(nproc)" \
  -e SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  -e FLASK_ENV=production \
  -v loanwise-data:/data \
  loanwise
```

The image runs as a non-root user, serves via gunicorn, and includes a
healthcheck. `/data` is the volume mount — the database lives there so it
survives a container replacement.

Create the first administrator:

```bash
docker exec -it loanwise flask --app app create-admin \
  --user-id admin --email admin@example.com
```

Behind a reverse proxy, set `TRUSTED_PROXY=1` so rate limiting reads the real
client IP. Do **not** set it when the app is directly exposed: `X-Forwarded-For`
is trivially spoofable, and trusting it lets a caller mint a fresh rate-limit
bucket per request.

### Platforms this image runs on unchanged

Fly.io, Render, Railway, Google Cloud Run, AWS App Runner, DigitalOcean App
Platform, or any Kubernetes cluster. All of them offer a persistent volume,
which is the only thing the app needs beyond a Python runtime.

On **Cloud Run** specifically, note that instances are ephemeral and the local
filesystem is not shared between them — treat it like the serverless case below
and move to Postgres before running more than one instance.

---

## Why not serverless

`vercel.json` used to be in this repository. It was removed, because the
application is stateful and the platform cannot hold that state.

### What would work

Anonymous assessment — the form, the result page, factor explanations,
guidance, affordability, `/api/assess`, `/api/compare`, `/api/model-card`.

### What cannot

Accounts, sign-in, saved applications, and the admin view. Three reasons, none
fixable by configuration:

1. **The filesystem is read-only outside `/tmp`.** SQLite cannot open a
   writable database anywhere else.
2. **`/tmp` does not persist.** It is destroyed on cold start.
3. **`/tmp` is not shared between instances.** A user can register and be
   signed out moments later because the next request landed on a different
   instance whose database has never heard of them.

A config that silently loses user data is worse than no config, so the honest
move was to delete it rather than ship it with a caveat in the README.

`loanapp/config.py` still detects `VERCEL` and `AWS_LAMBDA_FUNCTION_NAME`,
defaults `DATABASE_PATH` to `/tmp`, and logs a warning — so anyone who deploys
there deliberately gets a working anonymous-assessment demo and an explicit
statement that nothing persists. It is just not a path this project recommends
or maintains config for.

The 50 MB lambda bundle limit is also tight with pandas and scikit-learn.

---

## Postgres migration

SQLite is the right choice for a single container at this scale. Move when any
of these becomes true:

- More than one application instance needs to write
- Concurrent writes are producing `database is locked` errors
- You need point-in-time recovery or a managed backup story
- The applications table passes a few million rows

The work is contained: `loanapp/db.py` holds every SQL statement, all
parameterised, all behind two repository classes. The blueprints never touch
SQL. A port means rewriting that one module and swapping `connect()` for a
connection pool — the domain and HTTP layers are untouched.

The schema is deliberately narrow (JSON payload plus indexed columns) precisely
so this port stays small.

---

## Model artifacts

The trained bundle (`models/loan_model.joblib`, ~1 MB) is **committed to the
repository**. This is a deliberate trade:

- **For:** the app works immediately after clone, and CI can verify that the
  committed bundle still matches the feature contract in the code.
- **Against:** a binary in git history, and the bundle must be regenerated
  whenever `FEATURE_COLUMNS` changes.

CI guards the failure mode. If you change the feature set without retraining,
`test_registry.py` fails and the `train` job fails.

At larger scale, move to a model registry (MLflow, S3 with versioned keys) and
fetch at startup. The bundle already carries a version string and metadata, so
the loader would barely change.

**Security note:** `joblib.load` executes pickled code. Only point `MODEL_DIR` at
artifacts you produced yourself or fetched over a channel you trust.

---

## Operations

### Health

`GET /api/health` returns:

```json
{
  "status": "ok",
  "app_version": "2.1.0",
  "model_status": "ready",
  "model": {
    "version": "…",
    "calibrated": true,
    "metrics": { … },
    "runtime_versions": { "scikit_learn": "1.9.1", "numpy": "2.4.6" }
  }
}
```

It returns **200 even when no model is loaded** (`"status": "degraded"`). That is
deliberate: the process is alive, everything except assessment works, and
restarting will not conjure a model. Use `status`, not the HTTP code, to alert on
model availability.

`model_status` values:

| Value | Meaning | Action |
|---|---|---|
| `ready` | Serving normally | — |
| `unavailable` | No bundle found | Run the trainer, redeploy |
| `contract_mismatch` | Bundle disagrees with the code | **Deployment error.** Retrain and redeploy. |

`contract_mismatch` covers two cases, and the error in the startup log says
which:

- **Feature mismatch** — code whose `FEATURE_COLUMNS` differ from the deployed
  bundle's.
- **Library mismatch** — the bundle was pickled under a different minor version
  of scikit-learn or numpy. `GET /api/health` publishes the bundle's
  `runtime_versions` so you can compare without shell access.

Either way the app refuses to serve rather than scoring a mis-aligned vector or
calling into an incompatible pickle. Both are fixed the same way: retrain
(`make train`), commit the bundle, redeploy.

### Upgrading dependencies

```bash
pip-audit -r requirements.txt     # what needs to move, and why
# edit the pin, then:
make train                        # regenerate the bundle under the new libraries
make check                        # lint + tests + audit
```

The bundle must be regenerated whenever scikit-learn or numpy changes minor
version. CI enforces this: the `train` job fails if the committed bundle cannot
be loaded and scored by the pinned library set.

Pins are exact rather than ranges because the bundle is a pickle. The cost is
manual upgrades; the mitigation is the weekly `pip-audit` run in CI, so a new
advisory surfaces as a failing scheduled build rather than waiting for someone
to look.

`.github/dependabot.yml` automates the part that is safe to automate. It
proposes grouped PRs for the web and dev dependencies, and patch releases for
everything — but it deliberately does **not** propose minor or major upgrades
of scikit-learn or numpy. Those invalidate the bundle, so a bot PR could only
ever be red until a human runs `make train`. Patch releases are proposed,
because those are the security fixes and the load-time guard tolerates patch
drift for exactly that reason.

### What to monitor

- `/api/health` → `status` and `model_status`
- `/admin/stats` → outcome mix. A sharp shift with no model change usually means
  the input population changed, which is worth investigating.
- Log lines matching `rate limit hit` — sustained hits suggest scraping or
  credential stuffing.
- Log lines matching `admin access denied`.

### Backups

```bash
docker exec loanwise sqlite3 /data/loanwise.db ".backup '/data/backup.db'"
docker cp loanwise:/data/backup.db ./backup-$(date +%F).db
```

Use `.backup`, not a file copy — the database runs in WAL mode and a naive copy
can capture a torn state.

### Scaling

One container handles this workload comfortably; scoring a calibrated linear
model is sub-millisecond and the model is loaded once per worker.

Before adding instances, move to Postgres. SQLite has a single writer, and two
instances pointed at the same volume will produce lock contention.
