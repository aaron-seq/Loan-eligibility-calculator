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
| `TRUSTED_PROXY` | only behind a proxy | Boolean. Makes rate limiting trust `X-Forwarded-For` |
| `TRUSTED_PROXY_DEPTH` | no | Proxies in front of the app, default `1`. Read the client IP this many hops from the right of `X-Forwarded-For` |
| `WEB_CONCURRENCY` | no | gunicorn worker processes (Docker default 2). **Set to the CPU count**: throughput scales linearly with processes, not threads — see [`BENCHMARKS.md`](BENCHMARKS.md) |
| `RATE_LIMIT_PREDICT` | no | `<requests>/<seconds>`, default `30/3600`. Raise it if many users share one IP (corporate NAT) |
| `RATE_LIMIT_LOGIN` / `RATE_LIMIT_REGISTER` | no | Same format; defaults `10/300` and `5/3600`. A malformed value stops the app from starting |
| `SESSION_COOKIE_SECURE` | no | Boolean. Defaults to true in production |
| `ANNUAL_INTEREST_RATE` | no | Fraction, e.g. `0.105`. Affordability reference rate; does not change the model's estimate — see [Troubleshooting](#changing-annual_interest_rate-does-not-move-the-probability) |
| `APPROVE_THRESHOLD` / `BORDERLINE_THRESHOLD` | no | Decision bands, `0 < borderline < approve < 1` |

Booleans accept `true`/`false`, `1`/`0`, `yes`/`no` or `on`/`off`, in any case.
Anything else stops the app at startup. Unrecognised values used to be
treated as false, so `SESSION_COOKIE_SECURE=ture` quietly sent the session
cookie over plain HTTP, and `TRUSTED_PROXY=false` — read as "any non-empty
string" — *enabled* trust in the spoofable header.

Requires Python 3.11 or newer; numpy, pandas and scikit-learn set that floor.

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

Use a Linux host that supports this container with a writable persistent local
volume at `/data`. Check storage durability, SQLite locking semantics and
backup/restore on the actual host before deploying. A provider accepting a
container image does not establish those properties. Managed multi-instance
or ephemeral hosting requires a network database first; see [#24](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/24).

No hosted deployment or disaster-recovery exercise was performed in the
[2026-10-23 review](engineering-evidence.md).

### Browser policy and TLS

The application sets CSP (`default-src 'none'`, same-origin styles/images,
no scripts, same-origin form targets, no frames or base URLs),
`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and
`Referrer-Policy: same-origin` on HTML, JSON, errors, redirects and static
responses. Keep assets in the committed stylesheet; inline scripts or styles
will be blocked. See [Flask's security guidance](https://flask.palletsprojects.com/en/stable/web-security/).
Dynamic responses also send `Cache-Control: no-store`; this does not erase
previously cached copies or control screenshots/downloads. Static CSS retains
its own cache policy.

Terminate HTTPS at a trusted proxy and configure HSTS there after verifying
the domain's TLS setup. The app does not send HSTS for local HTTP development.

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
  "app_version": "2.2.0",
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
| `load_failed` | Bundle exists but cannot be read — corrupt, truncated or unreadable | Retrain and redeploy; check file permissions |

Whenever the model is not `ready`, the response carries a `model_error` field
with the same text as the startup log, so you can diagnose without shell
access.

`contract_mismatch` covers two cases, and `model_error` says which:

- **Feature mismatch** — code whose `FEATURE_COLUMNS` differ from the deployed
  bundle's. The message lists both column sets.
- **Library mismatch** — the bundle was pickled under a different minor version
  of scikit-learn or numpy. The message names the bundle's versions and the
  running ones; a healthy response publishes them as `model.runtime_versions`.

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
docker exec loanwise python -c "import sqlite3; src = sqlite3.connect('/data/loanwise.db'); dst = sqlite3.connect('/data/backup.db'); src.backup(dst); dst.close()"
docker cp loanwise:/data/backup.db ./backup-$(date +%F).db
```

Use SQLite's backup API, not a file copy — the database runs in WAL mode and a
naive copy can capture a torn state. It goes through Python because the slim
base image does not ship the `sqlite3` command-line tool.

### Schema v2 upgrade

Application 2.2.0 upgrades schema v1 automatically at startup in one
`BEGIN IMMEDIATE` transaction. It makes scores nullable and removes model
claims only from historical assessments already marked `out_of_range`.
Supported assessments, owners, identifiers (including the deleted-ID high-water
mark), timestamps and model versions are preserved; history is not rescored.
The upgrade follows SQLite's [create/copy/drop/rename procedure](https://www.sqlite.org/lang_altertable.html).

Before rollout, stop application writers, make a SQLite backup as above, and
validate the upgrade on a disposable copy with the pinned runtime. The table
copy needs free disk space and holds a write lock; production database size and
upgrade duration have not been measured. Malformed historical assessment JSON
aborts startup and rolls the schema/data changes back. Repair a backup copy
after inspecting the affected record; do not delete real history to force startup.

The new API has nullable probabilities and an `unavailable` band. Upgrade
consumers first; see [API compatibility notes](API.md#out-of-range-applications).
Older application code cannot safely use schema v2. To roll back, stop writers
and restore the **pre-upgrade backup and previous application together** in a
clean data location. Do not overwrite a live WAL database. Writes made after
the backup require a separate reconciliation decision. Newer schema versions
are refused rather than silently downgraded.

Disposable tests cover supported/unsupported history, constraints, repeated and
concurrent startup, failed-upgrade rollback and future-version refusal in
[test_migrations.py](../tests/test_migrations.py). Actual-host recovery and
deployment remain unverified.

### Scaling

Scale up before scaling out. A full assessment costs about 33 ms of CPU and
holds the GIL, so throughput tracks worker *processes*: on 4 CPUs, measured
20.9 / 41.2 / 86.3 req/s at 1 / 2 / 4 workers ([`BENCHMARKS.md`](BENCHMARKS.md)).
Set `WEB_CONCURRENCY` to the CPU count and give the container more CPUs before
anything else. Each worker loads its own copy of the model.

Before adding instances, move to Postgres. SQLite has a single writer, and two
instances pointed at the same volume will produce lock contention.

---

## Troubleshooting

Each entry is a failure this project has actually hit or been designed
against: the symptom you see, why it happens, and the fix.

### The app exits at startup with a `ConfigError`

Configuration is validated at boot, deliberately, so a bad value fails once
and loudly instead of on some later request. The message names the variable:

| Message begins | Fix |
|---|---|
| `SECRET_KEY is required in production` | Set `SECRET_KEY`; the message prints a command that generates one |
| `SECRET_KEY must be at least 32 characters` | Generate a longer key |
| `FLASK_ENV must be production, development or testing` | Fix the spelling |
| `RATE_LIMIT_… must be <requests>/<seconds>` | Use the form `30/3600`; both numbers positive |
| `… must be true or false` | Use one of the boolean spellings [above](#required-configuration) |
| `TRUSTED_PROXY_DEPTH counts proxies…` | Use 1 or more |
| `SESSION_COOKIE_SAMESITE must be Lax, Strict or None` | Fix the spelling |
| `SESSION_COOKIE_SAMESITE=None requires SESSION_COOKIE_SECURE=true` | Set both, or use `Lax` |
| `Require 0 < BORDERLINE_THRESHOLD < APPROVE_THRESHOLD < 1` | Thresholds are fractions, borderline below approve |
| `ANNUAL_INTEREST_RATE is a fraction` | Write `0.105`, not `10.5` |

### `/api/health` says `degraded`, and assessments return 503

Read `model_status` and `model_error` in the same response — see
[Health](#health). Every case is fixed by producing a bundle under the
libraries you are running: `make train`, commit `models/`, redeploy.

The Docker image trains during the build, against the exact libraries it
ships, so a `contract_mismatch` should be impossible inside it. If you see
one, something is replacing the built bundle — most likely a volume mounted
over `MODEL_DIR` that holds an older one.

### A setting in `.env` is ignored

A real environment variable of the same name wins over `.env`, by design: a
stray file must never replace a production secret. Check what the process
actually sees (`docker exec loanwise env`, or your platform's settings). Also
check the location: the file is read from next to `app.py`, not from the
current directory. `.dockerignore` keeps it out of the image, so containers
never read one.

### Users are randomly signed out

`SECRET_KEY` is unset in development, so every worker generated its own
ephemeral key and a session signed by one worker is rejected by the next. The
startup log says so. Set `SECRET_KEY`. Production refuses to start without it.

### A whole office gets 429 Too Many Requests

Everyone behind one NAT shares one IP, and the default is 30 assessments per
hour per IP. Raise `RATE_LIMIT_PREDICT`. If the app sits behind your own
reverse proxy, also set `TRUSTED_PROXY=true` — otherwise every request
appears to come from the proxy and *all* users share a single bucket.

### Rate limits seem several times too lenient

The limiter is per process, so N workers allow roughly N times the configured
limit. That is a known limitation, tracked in
[#18](https://github.com/aaron-seq/Loan-eligibility-calculator/issues/18).
Size limits with the worker count in mind.

### Pages render unstyled

`static/css/app.css` is missing or out of date. It is a committed build of
`static/src/tailwind.css`; regenerate it with `make css`. No Node is needed —
the Tailwind standalone binary is installed through pip.

Use `make css` rather than calling `pytailwindcss` directly. The Makefile pins
Tailwind v3.4.19 through `TAILWINDCSS_VERSION`; without that variable the
tool downloads the latest release — v4, a different engine that generates a
different stylesheet from the same templates.

### A Tailwind class added to a template has no effect

Same cause: the stylesheet only contains classes that were present when it
was built. Run `make css` and commit the result. CI fails if the committed
file does not match a fresh build, so this cannot reach `main` unnoticed.

### The container is unhealthy or restart-loops

Check the log for `Configuration error:` first. A configuration error stops
gunicorn with exit status 3 and `Reason: Worker failed to boot.`, so under a
restart policy it shows up as a restart loop (Kubernetes: `CrashLoopBackOff`).
The fix is the variable the message names — see the
[startup table](#the-app-exits-at-startup-with-a-configerror).

If the process is up but unhealthy: the healthcheck calls
`http://127.0.0.1:${PORT}/api/health`, so confirm that `PORT` matches what the
platform routes to. A missing model does *not* fail the healthcheck.

Before this was fixed, a configuration error exited the *worker* with status
1, which gunicorn treats as an ordinary crash: it respawned the worker every
two seconds indefinitely, with the master alive and the port open. The
container looked "running" and served nothing. If you see that pattern, you
are running an older build.

### `database is locked`

SQLite allows one writer at a time, and each connection waits up to 30
seconds for its turn, so seeing this at all means real contention. The usual
causes are two instances sharing one database file, or a database on a
network filesystem (NFS, SMB, some cloud volumes), where WAL mode does not
work. Use a local volume and a single instance, or move to Postgres — see
[Postgres migration](#postgres-migration).

### Changing `ANNUAL_INTEREST_RATE` does not move the probability

By design. The model was trained on features computed at a specific rate,
recorded in the bundle, and it always scores at that rate. Feeding it
features computed at a different rate would shift its inputs away from what
it learned. The setting does change the affordability panel and the EMI
rules, which are formulas rather than learned parameters. To change the
model's rate, retrain.

### The accessibility tests are all skipped

They need a Chromium that Playwright can drive. Run
`playwright install chromium`, or point `PLAYWRIGHT_CHROMIUM_EXECUTABLE` at
an existing binary. CI sets `A11Y_REQUIRED=1`, which turns a skip into a
failure, so the audit cannot pass by not running.
