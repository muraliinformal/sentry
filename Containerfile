# Copyright 2026 Sentry Object Intelligence contributors
# SPDX-License-Identifier: Apache-2.0

FROM mirror.gcr.io/library/python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    SENTRY_DB_PATH=/app/data/tracking_database.db \
    SENTRY_DATASET_DIR=/app/data/dataset \
    SENTRY_VEHICLE_MODEL_PATH=/app/models/yolo26m.pt \
    SENTRY_FACE_MODEL_PATH=/app/models/yolo26x-face.pt \
    SENTRY_INSIGHTFACE_ROOT=/app/insightface_models \
    SENTRY_INSIGHTFACE_MODEL_NAME=buffalo_l \
    SENTRY_DEEPFACE_MODEL_NAME=VGG-Face \
    DEEPFACE_HOME=/app/.deepface_home \
    MPLCONFIGDIR=/app/.matplotlib_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

COPY app.py pipeline.py README.md /app/
COPY models/ /app/models/
COPY insightface_models/ /app/insightface_models/
COPY .deepface_home/ /app/.deepface_home/

RUN mkdir -p /app/data/dataset /app/.deepface_home /app/.matplotlib_cache

EXPOSE 8501
VOLUME ["/app/data"]

CMD ["streamlit", "run", "app.py"]
