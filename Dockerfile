# Production image. This is the recommended way to run the app — see
# docs/DEPLOYMENT.md for why a container beats a serverless target for a
# stateful application.
#
# The model is trained during the build rather than copied from the repository.
# scikit-learn pickles are not portable across library versions, so building
# the artifact against the exact pins installed in *this* image is what
# guarantees it stays loadable. The committed bundle in models/ still serves
# local development and the CI contract check; the image simply does not depend
# on it. loanapp/ml/registry.py refuses a bundle whose recorded scikit-learn or
# numpy version does not match the running one, so a mismatch here fails loudly
# at startup instead of at the first request.

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Train against the versions just installed, so the pickle matches the runtime.
COPY loanapp/ ./loanapp/
COPY data/train.csv ./data/train.csv
# The site-packages path is derived rather than hardcoded: writing
# python3.13/site-packages couples this line to the base image tag above, and
# bumping that tag would break the build in a way that looks unrelated.
RUN PYTHONPATH="$(ls -d /install/lib/python*/site-packages)" \
    FLASK_ENV=development \
    python -m loanapp.ml.train --model-dir /build/models --no-docs


FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/models/ ./models/

COPY loanapp/ ./loanapp/
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/
COPY app.py ./

# Run unprivileged. /data is the mount point for the SQLite volume, so the
# database survives a container restart; without a volume, state is lost on
# redeploy.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

ENV FLASK_ENV=production \
    DATABASE_PATH=/data/loanwise.db \
    MODEL_DIR=/app/models \
    PORT=8000

EXPOSE 8000

# The health endpoint returns 200 even with no model loaded — the process is
# alive, everything except assessment works, and restarting will not conjure a
# model. Deliberate; see loanapp/blueprints/api.py.
#
# ${PORT}, not a hardcoded 8000: the server binds to ${PORT}, so with any other
# value a fixed port made the check fail forever and the orchestrator
# restart-looped a healthy container. Shell form, so the variable expands.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT}/api/health').status==200 else 1)"

# Worker count comes from WEB_CONCURRENCY, the convention gunicorn and most
# hosting platforms use; set it to the number of CPUs. Measured with
# scripts/benchmark.py on 4 CPUs: 20.9 / 41.2 / 86.3 req/s at 1 / 2 / 4
# workers — linear, because scoring is CPU-bound and holds the GIL, so extra
# *threads* add no throughput and extra *processes* do. The previous fixed
# "--workers 2" left half of a 4-CPU host idle.
#
# Threads stay at 2 only so a slow client cannot block a worker outright.
# Each worker loads its own copy of the model (small for a linear model), and
# multi-worker startup is safe: loanapp/db.py sets WAL once at init.
ENV WEB_CONCURRENCY=2
CMD ["sh", "-c", "gunicorn --workers ${WEB_CONCURRENCY} --threads 2 --timeout 60 --bind 0.0.0.0:${PORT} app:app"]
