FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIKEFLOW_ROOT=/app \
    BIKEFLOW_MODEL_PATH=/models/model.joblib

WORKDIR /app

RUN addgroup --system bikeflow && adduser --system --ingroup bikeflow bikeflow

COPY pyproject.toml README.md params.yaml ./
COPY requirements/runtime-py311.lock ./requirements/runtime-py311.lock
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install --constraint requirements/runtime-py311.lock .

USER bikeflow
EXPOSE 8000

CMD ["uvicorn", "bikeflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
