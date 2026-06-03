# Copyright 2026 Sentry Object Intelligence contributors
# SPDX-License-Identifier: Apache-2.0

FROM mirror.gcr.io/library/python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

COPY requirements-container.txt /tmp/requirements-container.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-container.txt


FROM mirror.gcr.io/library/python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    SENTRY_DB_PATH=/app/data/tracking_database.db \
    SENTRY_DATASET_DIR=/app/data/dataset \
    SENTRY_VEHICLE_MODEL_PATH=/app/models/yolo26m.pt \
    SENTRY_FACE_MODEL_PATH=/app/models/yolo26x-face.pt \
    SENTRY_INSIGHTFACE_ROOT=/app/insightface_models \
    SENTRY_INSIGHTFACE_MODEL_NAME=buffalo_l \
    MPLCONFIGDIR=/app/.matplotlib_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY app.py pipeline.py README.md /app/
COPY models/ /app/models/
COPY insightface_models/ /app/insightface_models/

RUN mkdir -p /app/data/dataset /app/.matplotlib_cache

EXPOSE 8501
VOLUME ["/app/data"]

CMD ["streamlit", "run", "app.py"]
