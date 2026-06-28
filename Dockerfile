FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Build deps for pandas/numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the anomaly-detection model at build time so the image ships ready.
# (Falls back to lazy training at first request if this is ever skipped.)
RUN python -m app.ml.train

# Railway provides $PORT at runtime; default to 8001 for local `docker run`.
EXPOSE 8001
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}
