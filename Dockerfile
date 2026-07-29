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
ENV PYTHONPATH=/install/lib/python3.13/site-packages
RUN FLASK_ENV=development python -m loanapp.ml.train --model-dir /build/models --no-docs


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
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# Two workers: the model is loaded per worker, so memory scales with this.
# Multi-worker startup is safe — loanapp/db.py sets the WAL journal mode once
# at init rather than per connection, which is what made concurrent workers
# collide on the exclusive lock.
CMD ["sh", "-c", "gunicorn --workers 2 --threads 4 --timeout 60 --bind 0.0.0.0:${PORT} app:app"]
