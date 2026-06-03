# Sentry Object Intelligence

Copyright 2026 Sentry Object Intelligence contributors. Licensed under the
Apache License, Version 2.0. See `LICENSE`.

Production-ready face and vehicle tracking for RTSP streams, local videos, and
static images. The pipeline uses configurable YOLO weights, InsightFace
`buffalo_l` embeddings for face recognition, DeepFace fallback verification,
EasyOCR registration extraction, HSV vehicle color classification, and SQLite
persistence.

## Runtime Structure

```text
dataset/                # Legacy/on-disk crop storage when file retention is enabled
tracking_database.db    # SQLite database
pipeline.py             # Core processing logic
app.py                  # Streamlit Web UI dashboard
```

Existing auxiliary files or directories in this checkout are not required by
the application.

## Model Configuration

Provide model paths in the Streamlit sidebar or set environment variables
before running. If the sidebar fields are left blank, the app automatically
uses local `.pt` files from `models/` for both detector slots. If only one
`.pt` file exists, it is used for both vehicle and people/face detection.

> **Model redistribution note:** this public repository intentionally does not
> include pretrained model weights. Place your own licensed model files under
> `models/`, `insightface_models/`, and/or `.deepface_home/` before running or
> building a fully self-contained image. See `THIRD_PARTY_NOTICES.md` for
> dependency, model, and biometric/privacy notices.

```bash
export SENTRY_VEHICLE_MODEL_PATH="/absolute/path/to/vehicle-detector.pt"
export SENTRY_FACE_MODEL_PATH="/absolute/path/to/face-detector.pt"
export SENTRY_INSIGHTFACE_MODEL_NAME="buffalo_l"
export SENTRY_INSIGHTFACE_ROOT="$HOME/.insightface"
export SENTRY_DEEPFACE_MODEL_NAME="VGG-Face"
```

If `buffalo_l` exists under `$HOME/.insightface/models/buffalo_l`, the app uses
InsightFace automatically for face matching. If InsightFace is unavailable or
cannot extract an embedding, it falls back to DeepFace. If
`SENTRY_DEEPFACE_MODEL_NAME` is not set, the fallback model is `VGG-Face`.

By default, new crop images are stored as compressed JPEG BLOBs in SQLite and
not written to `dataset/`. Set `PipelineConfig(keep_image_files=True)` if you
also want on-disk crop files for debugging or external review.

## Install

```bash
pip install streamlit ultralytics opencv-python numpy easyocr insightface onnxruntime deepface
```

## Run Dashboard

```bash
streamlit run app.py
```

## Run Programmatically

```python
import pipeline

config = pipeline.PipelineConfig.with_model_paths(
    vehicle_model_path="/absolute/path/to/vehicle-detector.pt",
    face_model_path="/absolute/path/to/face-detector.pt",
    deepface_model_name="VGG-Face",
)

summary = pipeline.run_pipeline(
    "rtsp://user:password@camera/live",
    source_type="live",
    camera_location="Front Door",
    config=config,
)
print(summary)
```

For local files, pass `source_type="video"` or `source_type="image"`.

## Database

The `tracked_entities` table is initialized automatically with:

- `id`
- `category`
- `custom_name`
- `reg_number`
- `color`
- `vehicle_type`
- `make_model`
- `camera_location`
- `image_path`
- `last_seen_timestamps`
- `image_blob`
- `image_mime`
- `face_embedding`

`last_seen_timestamps` is a JSON array string containing up to the latest five
unique detections for each matched entity.

## Verification

```bash
python -m py_compile pipeline.py app.py
```

## Container build and portable archive

The `Containerfile` can package Python dependencies plus any locally supplied
detector, InsightFace, and DeepFace model assets from `models/`,
`insightface_models/`, and `.deepface_home/.deepface/weights/`. Runtime data is
intentionally kept outside the image through `/app/data`.

The `.containerignore`/`.dockerignore` files exclude `dataset/`,
`tracking_database.db`, generated archives, runtime caches, and common
image/video extensions so captured media is not baked into the container.

Before building a deployable image, add your own licensed model files locally,
for example:

```text
models/
  yolo26m.pt
  yolo26x-face.pt
insightface_models/
  models/buffalo_l/*.onnx
.deepface_home/
  .deepface/weights/vgg_face_weights.h5
```

```bash
podman build -t sentry-object-intelligence:latest -f Containerfile .
podman save -o sentry-object-intelligence.tar sentry-object-intelligence:latest
```

Run the image with a host-mounted data directory for the SQLite database and any
runtime crop/output files:

```bash
mkdir -p sentry-data
podman run --rm -p 8501:8501 -v "$(pwd)/sentry-data:/app/data:Z" sentry-object-intelligence:latest
```

## GitHub publishing and licensing

This project is licensed under the Apache License 2.0. See `LICENSE` and
`NOTICE`.

For safer public publishing, `.gitignore` excludes local model weights,
container tar archives, runtime databases, datasets, captured media, caches, and
logs. Do not publish pretrained weights, generated container images, or captured
media unless you have verified that redistribution is permitted by the relevant
licenses and privacy laws.

See `THIRD_PARTY_NOTICES.md` for third-party dependency, model-weight,
container-distribution, and biometric/privacy cautions. `THIRDPART_NOTICE` is
also provided as a compatibility pointer to that canonical notice file.
