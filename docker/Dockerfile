FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost / matplotlib / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 -r requirements.txt

COPY src/ src/
COPY models/ models/

# Model artifacts are NOT baked into the image (keeps images small and
# avoids stale models). Mount your local models/ and logs/ folders as
# volumes at runtime — see docker/docker-compose.yml.
RUN mkdir -p models logs

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
