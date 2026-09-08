FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIKEFLOW_ROOT=/app \
    BIKEFLOW_MODEL_PATH=/models/model.joblib

WORKDIR /app

COPY pyproject.toml README.md params.yaml ./
COPY requirements/runtime-py311.lock ./requirements/runtime-py311.lock
COPY src ./src

RUN python -m pip install --upgrade pip

FROM base AS training

RUN python -m pip install --constraint requirements/runtime-py311.lock \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --constraint requirements/runtime-py311.lock ".[ml,mlp]"

CMD ["python", "-m", "bikeflow.ml", "train", "--no-figures"]

FROM base AS runtime

RUN addgroup --system bikeflow && adduser --system --ingroup bikeflow bikeflow && \
    python -m pip install --constraint requirements/runtime-py311.lock \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --constraint requirements/runtime-py311.lock ".[mlp]"

USER bikeflow
EXPOSE 8000

CMD ["uvicorn", "bikeflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
