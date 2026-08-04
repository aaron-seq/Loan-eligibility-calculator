# Deployment image for platforms with a real filesystem (Railway, Render,
# Fly.io, any VPS). This app keeps state in SQLite, so it needs a writable
# persistent volume — see README "Deployment".
#
# The model artefacts are trained during the build rather than committed:
# scikit-learn pickles are not portable across library versions, so building
# them against the exact pinned versions in this image is what keeps them
# loadable.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Train against the installed versions, so the pickles match the runtime.
COPY models/ ./models/
COPY data/train.csv ./data/train.csv
ENV PYTHONPATH=/install/lib/python3.11/site-packages
RUN python models/model_trainer.py


FROM python:3.11-slim

# Non-root: nothing here needs privileges.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/models/*.pkl ./models/
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/model_trainer.py ./models/

# SQLite lives on a mounted volume; without one, state is lost on redeploy.
RUN mkdir -p /data && chown appuser:appuser /data
VOLUME ["/data"]
ENV DATABASE_PATH=/data/loan_prediction.db \
    MODEL_PATH=/app/models \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

# SECRET_KEY and ADMIN_PASSWORD must be supplied at runtime; the app refuses
# to start without SECRET_KEY outside development.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", \
     "--access-logfile", "-", "app:app"]
