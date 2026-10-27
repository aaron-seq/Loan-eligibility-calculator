# Train inside the image so the bundle matches its pinned ML libraries.
# The committed bundle remains available for local development and CI.

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY loanapp/ ./loanapp/
COPY data/train.csv ./data/train.csv
# Derive site-packages from the installed Python version.
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
COPY data/train.csv ./data/train.csv
COPY app.py ./
COPY LICENSE THIRD_PARTY_NOTICES.md ./

# Keep SQLite state on the persistent /data volume; run unprivileged.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

ENV FLASK_ENV=production \
    DATABASE_PATH=/data/loanwise.db \
    MODEL_DIR=/app/models \
    PORT=8000

EXPOSE 8000

# Liveness remains 200 when the model is degraded; monitor model_status too.
# Shell form expands the same PORT used by Gunicorn.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT}/api/health').status==200 else 1)"

# Workers trade throughput for model memory; measure on the deployment host.
ENV WEB_CONCURRENCY=2
CMD ["sh", "-c", "gunicorn --workers ${WEB_CONCURRENCY} --threads 2 --timeout 60 --bind 0.0.0.0:${PORT} app:app"]
