# Copyright 2026 Sentry Object Intelligence contributors
# SPDX-License-Identifier: Apache-2.0

"""Core object tracking and intelligence pipeline.

The module is intentionally configuration-driven: model weight paths are passed
through ``PipelineConfig`` or environment variables instead of being embedded in
the inference code. Heavy ML libraries are lazy-loaded and cached so the
dashboard and database can start even when model runtimes are not yet present.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime dependency
    np = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("SENTRY_DATASET_DIR", str(BASE_DIR / "dataset"))).expanduser()
MODELS_DIR = BASE_DIR / "models"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib_cache"
DB_PATH = Path(os.environ.get("SENTRY_DB_PATH", str(BASE_DIR / "tracking_database.db"))).expanduser()
TABLE_NAME = "tracked_entities"
DETECTION_EVENTS_TABLE = "detection_events"
UNKNOWN_REGISTRATION = "UNKNOWN_REG"
DEFAULT_INSIGHTFACE_MODEL_NAME = "buffalo_l"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

VEHICLE_MODEL_ENV = "SENTRY_VEHICLE_MODEL_PATH"
FACE_MODEL_ENV = "SENTRY_FACE_MODEL_PATH"
INSIGHTFACE_MODEL_ENV = "SENTRY_INSIGHTFACE_MODEL_NAME"
INSIGHTFACE_ROOT_ENV = "SENTRY_INSIGHTFACE_ROOT"
INSIGHTFACE_ENABLED_ENV = "SENTRY_INSIGHTFACE_ENABLED"
EASYOCR_GPU_ENV = "SENTRY_EASYOCR_GPU"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

# COCO vehicle class ids used by general YOLO checkpoints.
COCO_PERSON_CLASS_ID = 0
COCO_VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

LOGGER = logging.getLogger("sentry.pipeline")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class PipelineRuntimeError(RuntimeError):
    """Raised for recoverable runtime/configuration failures."""


ProgressCallback = Callable[[str], None]


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        LOGGER.debug("Progress callback failed", exc_info=True)


def _env_path(name: str) -> Path | None:
    raw_value = os.environ.get(name, "").strip()
    return Path(raw_value).expanduser() if raw_value else None


def _looks_like_model_weights(path: Path) -> bool:
    """Reject common accidental downloads such as HTML pages saved as .pt files."""

    try:
        with path.open("rb") as handle:
            prefix = handle.read(256).lstrip()
    except OSError:
        return False
    if not prefix:
        return False
    lowered = prefix[:64].lower()
    if lowered.startswith((b"<!doctype html", b"<html", b"<!doctype")):
        return False
    return True


def _model_candidates() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    candidates = []
    for path in sorted(MODELS_DIR.glob("*.pt")):
        if not path.is_file():
            continue
        if not _looks_like_model_weights(path):
            LOGGER.warning("Ignoring invalid model candidate %s; it does not look like PyTorch weights.", path)
            continue
        candidates.append(path)
    return candidates


def discover_default_model_path(*, role: str) -> Path | None:
    """Pick a sensible local YOLO weight when the user leaves model fields blank."""

    env_name = FACE_MODEL_ENV if role == "face" else VEHICLE_MODEL_ENV
    env_path = _env_path(env_name)
    if env_path is not None:
        return env_path

    candidates = _model_candidates()
    if not candidates:
        return None

    if role == "face":
        face_candidates = [path for path in candidates if "face" in path.stem.lower()]
        return face_candidates[0] if face_candidates else candidates[0]

    non_face_candidates = [path for path in candidates if "face" not in path.stem.lower()]
    return non_face_candidates[0] if non_face_candidates else candidates[0]


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _insightface_root() -> Path:
    return Path(os.environ.get(INSIGHTFACE_ROOT_ENV, "~/.insightface")).expanduser()


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for inference, persistence, and de-duplication."""

    db_path: Path = field(default_factory=lambda: DB_PATH)
    dataset_dir: Path = field(default_factory=lambda: DATASET_DIR)
    vehicle_model_path: Path | None = field(default_factory=lambda: discover_default_model_path(role="vehicle"))
    face_model_path: Path | None = field(default_factory=lambda: discover_default_model_path(role="face"))
    crop_padding_pixels: int = 18
    live_skip_frames: int = 12
    local_skip_frames: int = 3
    vehicle_confidence: float = 0.35
    face_confidence: float = 0.45
    face_match_max_exemplars_per_name: int = 4
    insightface_enabled: bool = field(default_factory=lambda: _env_bool(INSIGHTFACE_ENABLED_ENV, True))
    insightface_model_name: str = field(
        default_factory=lambda: os.environ.get(INSIGHTFACE_MODEL_ENV, DEFAULT_INSIGHTFACE_MODEL_NAME).strip()
        or DEFAULT_INSIGHTFACE_MODEL_NAME
    )
    insightface_root: Path = field(default_factory=_insightface_root)
    insightface_match_max_distance: float = 0.62
    insightface_ambiguity_margin: float = 0.12
    store_images_as_blobs: bool = True
    keep_image_files: bool = False
    crop_jpeg_quality: int = 82
    easyocr_languages: tuple[str, ...] = ("en",)
    easyocr_gpu: bool = field(default_factory=lambda: _env_bool(EASYOCR_GPU_ENV, False))
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 3.0
    max_timestamps_per_entity: int = 6
    deduplicate_unknown_registrations: bool = False

    @classmethod
    def with_model_paths(
        cls,
        *,
        vehicle_model_path: str | Path | None,
        face_model_path: str | Path | None,
    ) -> "PipelineConfig":
        """Build a config from UI/API supplied model paths."""

        return cls(
            vehicle_model_path=Path(vehicle_model_path).expanduser()
            if vehicle_model_path
            else discover_default_model_path(role="vehicle"),
            face_model_path=Path(face_model_path).expanduser()
            if face_model_path
            else discover_default_model_path(role="face"),
        )


@dataclass
class DetectionRuntimeState:
    """State that lives only for a single runner invocation."""

    vehicle_track_to_row: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredCrop:
    """Compressed crop storage payload."""

    image_path: str
    image_blob: bytes | None = None
    image_mime: str = "image/jpeg"


class ModelManager:
    """Thread-safe lazy loader/cache for YOLO, EasyOCR, and InsightFace."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._yolo_models: dict[Path, Any] = {}
        self._ocr_readers: dict[tuple[tuple[str, ...], bool], Any] = {}
        self._insightface_apps: dict[tuple[str, Path], Any] = {}

    def yolo(self, model_path: Path | None, *, role: str) -> Any:
        if model_path is None:
            raise PipelineRuntimeError(
                f"Missing {role} model path. Set it in PipelineConfig or the corresponding environment variable."
            )

        resolved = Path(model_path).expanduser().resolve()
        with self._lock:
            cached = self._yolo_models.get(resolved)
            if cached is not None:
                return cached

            if not resolved.exists():
                raise FileNotFoundError(f"{role.title()} model file not found: {resolved}")
            if not _looks_like_model_weights(resolved):
                raise PipelineRuntimeError(
                    f"{role.title()} model file is not valid PyTorch weights: {resolved}. "
                    "It may be an HTML/download page saved as .pt."
                )

            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency-specific
                raise ImportError("Missing dependency 'ultralytics'. Install it before inference.") from exc

            LOGGER.info("Loading %s YOLO model from %s", role, resolved)
            model = YOLO(str(resolved))
            self._yolo_models[resolved] = model
            return model

    def vehicle_model(self, config: PipelineConfig) -> Any:
        return self.yolo(config.vehicle_model_path, role="vehicle")

    def face_model(self, config: PipelineConfig) -> Any:
        return self.yolo(config.face_model_path, role="face")

    def easyocr_reader(self, config: PipelineConfig) -> Any:
        key = (tuple(config.easyocr_languages), bool(config.easyocr_gpu))
        with self._lock:
            cached = self._ocr_readers.get(key)
            if cached is not None:
                return cached

            try:
                import easyocr  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency-specific
                raise ImportError("Missing dependency 'easyocr'. Install it before inference.") from exc

            LOGGER.info("Initializing EasyOCR reader for languages=%s gpu=%s", key[0], key[1])
            reader = easyocr.Reader(list(key[0]), gpu=key[1])
            self._ocr_readers[key] = reader
            return reader

    def insightface_app(self, config: PipelineConfig) -> Any:
        model_name = config.insightface_model_name.strip()
        if not model_name:
            raise PipelineRuntimeError("Missing InsightFace model name.")

        root = Path(config.insightface_root).expanduser().resolve()
        key = (model_name, root)
        with self._lock:
            cached = self._insightface_apps.get(key)
            if cached is not None:
                return cached

            try:
                from insightface.app import FaceAnalysis  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency-specific
                raise ImportError("Missing dependency 'insightface'. Install it before inference.") from exc

            LOGGER.info("Loading InsightFace model %s from %s", model_name, root)
            app = FaceAnalysis(
                name=model_name,
                root=str(root),
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            if "recognition" not in getattr(app, "models", {}):
                raise PipelineRuntimeError(f"InsightFace model {model_name!r} did not load a recognizer.")
            self._insightface_apps[key] = app
            return app

DEFAULT_MODEL_MANAGER = ModelManager()


def _connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path | str = DB_PATH, dataset_dir: Path | str = DATASET_DIR) -> None:
    """Create storage directories and the unified SQLite schema."""

    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                custom_name TEXT NOT NULL,
                reg_number TEXT,
                color TEXT,
                vehicle_type TEXT,
                make_model TEXT,
                camera_location TEXT NOT NULL,
                image_path TEXT NOT NULL,
                last_seen_timestamps TEXT NOT NULL,
                image_blob BLOB,
                image_mime TEXT,
                face_embedding TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DETECTION_EVENTS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                image_blob BLOB,
                image_mime TEXT,
                detected_at TEXT NOT NULL,
                camera_location TEXT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES {TABLE_NAME}(id) ON DELETE CASCADE
            )
            """
        )
        _add_missing_columns(conn)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_category ON {TABLE_NAME}(category)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_reg_number ON {TABLE_NAME}(reg_number)")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{DETECTION_EVENTS_TABLE}_entity ON {DETECTION_EVENTS_TABLE}(entity_id)"
        )
        _seed_detection_events(conn)
        conn.commit()


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    }
    column_defs = {
        "category": "TEXT NOT NULL DEFAULT ''",
        "custom_name": "TEXT NOT NULL DEFAULT ''",
        "reg_number": "TEXT",
        "color": "TEXT",
        "vehicle_type": "TEXT",
        "make_model": "TEXT",
        "camera_location": "TEXT NOT NULL DEFAULT ''",
        "image_path": "TEXT NOT NULL DEFAULT ''",
        "last_seen_timestamps": "TEXT NOT NULL DEFAULT '[]'",
        "image_blob": "BLOB",
        "image_mime": "TEXT",
        "face_embedding": "TEXT",
    }
    for column_name, ddl in column_defs.items():
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column_name} {ddl}")

    existing_events = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({DETECTION_EVENTS_TABLE})").fetchall()
    }
    event_column_defs = {
        "image_blob": "BLOB",
        "image_mime": "TEXT",
    }
    for column_name, ddl in event_column_defs.items():
        if column_name not in existing_events:
            conn.execute(f"ALTER TABLE {DETECTION_EVENTS_TABLE} ADD COLUMN {column_name} {ddl}")


def _seed_detection_events(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        f"""
        SELECT id, image_path, image_blob, image_mime, camera_location, last_seen_timestamps
        FROM {TABLE_NAME}
        WHERE id NOT IN (SELECT entity_id FROM {DETECTION_EVENTS_TABLE})
        """
    ).fetchall()
    for row in rows:
        timestamps = _parse_timestamps(row["last_seen_timestamps"])
        detected_at = timestamps[-1] if timestamps else current_timestamp()
        conn.execute(
            f"""
            INSERT INTO {DETECTION_EVENTS_TABLE}
                (entity_id, image_path, image_blob, image_mime, detected_at, camera_location)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["id"]),
                str(row["image_path"]),
                row["image_blob"],
                row["image_mime"],
                detected_at,
                str(row["camera_location"]),
            ),
        )


def _name_key(name: str | None) -> str:
    return str(name or "").strip().casefold()


def _is_named_face_value(name: str | None) -> bool:
    value = str(name or "").strip()
    if not value or value == "Unidentified Face":
        return False
    return re.fullmatch(r"Face_\d{8}_\d{6}", value) is None


def _canonical_face_name(conn: sqlite3.Connection, requested_name: str, *, exclude_entry_id: int | None = None) -> str:
    cleaned_name = requested_name.strip()
    if not cleaned_name:
        return cleaned_name

    params: list[Any] = []
    exclude_clause = ""
    if exclude_entry_id is not None:
        exclude_clause = "AND id != ?"
        params.append(int(exclude_entry_id))
    params.append(_name_key(cleaned_name))

    row = conn.execute(
        f"""
        SELECT custom_name
        FROM {TABLE_NAME}
        WHERE category = 'Face'
          {exclude_clause}
          AND LOWER(TRIM(custom_name)) = ?
          AND custom_name IS NOT NULL
          AND TRIM(custom_name) != ''
          AND custom_name != 'Unidentified Face'
          AND custom_name NOT GLOB 'Face_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'
        GROUP BY custom_name
        ORDER BY COUNT(*) DESC, MIN(id) ASC
        LIMIT 1
        """,
        params,
    ).fetchone()
    return str(row["custom_name"]).strip() if row else cleaned_name


def fetch_entries(category: str | None = None, db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    """Fetch tracked entities, newest first."""

    init_database(db_path=db_path)
    sql = f"SELECT * FROM {TABLE_NAME}"
    params: list[Any] = []
    if category:
        sql += " WHERE category = ?"
        params.append(category)
    sql += " ORDER BY id DESC"
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def fetch_named_faces(db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    """Return named face groups for the people gallery."""

    init_database(db_path=db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                e.id,
                e.custom_name,
                ev.id AS detection_event_id,
                ev.detected_at
            FROM {TABLE_NAME} e
            LEFT JOIN {DETECTION_EVENTS_TABLE} ev ON ev.entity_id = e.id
            WHERE e.category = 'Face'
              AND e.custom_name IS NOT NULL
              AND TRIM(e.custom_name) != ''
              AND e.custom_name != 'Unidentified Face'
              AND e.custom_name NOT GLOB 'Face_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'
            """
        ).fetchall()

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            raw_name = str(row["custom_name"] or "").strip()
            key = _name_key(raw_name)
            if not key:
                continue
            person = grouped.setdefault(
                key,
                {
                    "custom_name": raw_name,
                    "entity_ids": set(),
                    "detection_event_ids": set(),
                    "last_detected_at": "",
                    "name_counts": {},
                    "first_ids": {},
                },
            )
            entity_id = int(row["id"])
            person["entity_ids"].add(entity_id)
            person["name_counts"][raw_name] = int(person["name_counts"].get(raw_name, 0)) + 1
            person["first_ids"][raw_name] = min(int(person["first_ids"].get(raw_name, entity_id)), entity_id)
            if row["detection_event_id"] is not None:
                person["detection_event_ids"].add(int(row["detection_event_id"]))
            detected_at = str(row["detected_at"] or "")
            if detected_at and detected_at > str(person["last_detected_at"]):
                person["last_detected_at"] = detected_at

        people: list[dict[str, Any]] = []
        for person in grouped.values():
            name_counts = dict(person.pop("name_counts"))
            first_ids = dict(person.pop("first_ids"))
            canonical_name = sorted(
                name_counts,
                key=lambda name: (-int(name_counts[name]), int(first_ids[name]), name),
            )[0]
            people.append(
                {
                    "custom_name": canonical_name,
                    "entity_count": len(person["entity_ids"]),
                    "detection_count": len(person["detection_event_ids"]),
                    "last_detected_at": person["last_detected_at"],
                }
            )

        people.sort(key=lambda person: str(person["custom_name"]).casefold())
        for person in people:
            timestamps = conn.execute(
                f"""
                SELECT ev.detected_at
                FROM {TABLE_NAME} e
                JOIN {DETECTION_EVENTS_TABLE} ev ON ev.entity_id = e.id
                WHERE e.category = 'Face' AND LOWER(TRIM(e.custom_name)) = ?
                ORDER BY ev.detected_at DESC, ev.id DESC
                LIMIT 2
                """,
                (_name_key(str(person["custom_name"])),),
            ).fetchall()
            person["last_detection_timestamps"] = [str(row["detected_at"]) for row in timestamps]
        return people


def fetch_face_detections_for_name(
    custom_name: str,
    db_path: Path | str = DB_PATH,
) -> list[dict[str, Any]]:
    """Return every stored detection image for a named person."""

    init_database(db_path=db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                ev.id AS detection_event_id,
                e.id AS entity_id,
                e.custom_name,
                ev.image_path,
                ev.image_blob,
                ev.image_mime,
                ev.detected_at,
                ev.camera_location,
                e.last_seen_timestamps
            FROM {TABLE_NAME} e
            JOIN {DETECTION_EVENTS_TABLE} ev ON ev.entity_id = e.id
            WHERE e.category = 'Face' AND LOWER(TRIM(e.custom_name)) = ?
            ORDER BY ev.detected_at DESC, ev.id DESC
            """,
            (_name_key(custom_name),),
        ).fetchall()
        return [dict(row) for row in rows]


def _fetch_face_match_candidates(
    conn: sqlite3.Connection,
    config: PipelineConfig,
    *,
    exclude_entry_id: int | None = None,
    limit_per_name: bool = True,
) -> list[sqlite3.Row]:
    """Return a bounded set of named face exemplars, preserving name case."""

    params: list[Any] = []
    exclude_clause = ""
    if exclude_entry_id is not None:
        exclude_clause = "AND id != ?"
        params.append(exclude_entry_id)

    rows = conn.execute(
        f"""
        SELECT id, custom_name, image_path, image_blob, face_embedding
        FROM {TABLE_NAME}
        WHERE category = 'Face'
          {exclude_clause}
          AND custom_name IS NOT NULL
          AND TRIM(custom_name) != ''
          AND custom_name != 'Unidentified Face'
          AND custom_name NOT GLOB 'Face_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'
        ORDER BY TRIM(custom_name) COLLATE BINARY, id DESC
        """,
        params,
    ).fetchall()

    if not limit_per_name:
        return list(rows)

    per_name_counts: dict[str, int] = {}
    selected: list[sqlite3.Row] = []
    max_per_name = max(1, int(config.face_match_max_exemplars_per_name))
    for row in rows:
        name = _name_key(str(row["custom_name"]))
        count = per_name_counts.get(name, 0)
        if count >= max_per_name:
            continue
        selected.append(row)
        per_name_counts[name] = count + 1
    return selected


def suggest_named_face_for_entry(
    entry_id: int,
    *,
    config: PipelineConfig | None = None,
    db_path: Path | str = DB_PATH,
) -> dict[str, Any] | None:
    """Suggest an existing named face for an unidentified review entry."""

    config = config or PipelineConfig(db_path=Path(db_path))
    init_database(db_path=db_path)
    if cv2 is None:
        return None

    with _connect(db_path) as conn:
        source = conn.execute(
            f"""
            SELECT id, image_path, image_blob
            FROM {TABLE_NAME}
            WHERE id = ? AND category = 'Face'
            """,
            (entry_id,),
        ).fetchone()
        if source is None:
            return None

        source_crop = _read_crop_from_row(source)
        if source_crop is None:
            return None

        insightface_match = _find_matching_face_by_insightface(
            conn,
            source_crop,
            DEFAULT_MODEL_MANAGER,
            config,
            exclude_entry_id=entry_id,
            limit_per_name=False,
        )
        if insightface_match is not None:
            conn.commit()
            return insightface_match

    return None


def update_entry(
    entry_id: int,
    *,
    custom_name: str | None = None,
    make_model: str | None = None,
    db_path: Path | str = DB_PATH,
) -> None:
    """Update editable fields using a parameterized SQL statement."""

    updates: list[str] = []
    params: list[Any] = []
    custom_name_param_index: int | None = None

    if custom_name is not None:
        cleaned_name = custom_name.strip()
        if cleaned_name:
            updates.append("custom_name = ?")
            custom_name_param_index = len(params)
            params.append(cleaned_name)

    if make_model is not None:
        updates.append("make_model = ?")
        params.append(make_model.strip())

    if not updates:
        return

    params.append(int(entry_id))
    with _connect(db_path) as conn:
        if custom_name_param_index is not None:
            row = conn.execute(
                f"SELECT category FROM {TABLE_NAME} WHERE id = ?",
                (int(entry_id),),
            ).fetchone()
            if row is not None and str(row["category"]) == "Face":
                params[custom_name_param_index] = _canonical_face_name(
                    conn,
                    str(params[custom_name_param_index]),
                    exclude_entry_id=int(entry_id),
                )
        conn.execute(f"UPDATE {TABLE_NAME} SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()


def _insert_detection_event(
    conn: sqlite3.Connection,
    *,
    entity_id: int,
    image_path: str,
    image_blob: bytes | None = None,
    image_mime: str | None = None,
    timestamp: str,
    camera_location: str,
) -> int:
    cursor = conn.execute(
        f"""
        INSERT INTO {DETECTION_EVENTS_TABLE}
            (entity_id, image_path, image_blob, image_mime, detected_at, camera_location)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (int(entity_id), image_path, image_blob, image_mime, timestamp, camera_location),
    )
    return int(cursor.lastrowid)


def delete_entry(
    entry_id: int,
    *,
    delete_image: bool = True,
    db_path: Path | str = DB_PATH,
    dataset_dir: Path | str = DATASET_DIR,
) -> bool:
    """Delete a tracked entity and optionally remove its saved crop image."""

    init_database(db_path=db_path, dataset_dir=dataset_dir)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT image_path FROM {DETECTION_EVENTS_TABLE} WHERE entity_id = ?
            UNION
            SELECT image_path FROM {TABLE_NAME} WHERE id = ?
            """,
            (int(entry_id), int(entry_id)),
        ).fetchall()
        row = conn.execute(
            f"SELECT id FROM {TABLE_NAME} WHERE id = ?",
            (int(entry_id),),
        ).fetchone()
        if row is None:
            return False

        image_paths = [Path(str(item["image_path"])) for item in rows]
        conn.execute(f"DELETE FROM {DETECTION_EVENTS_TABLE} WHERE entity_id = ?", (int(entry_id),))
        conn.execute(f"DELETE FROM {TABLE_NAME} WHERE id = ?", (int(entry_id),))
        conn.commit()

    if delete_image:
        resolved_dataset = Path(dataset_dir).expanduser().resolve()
        for image_path in image_paths:
            try:
                resolved_image = image_path.expanduser().resolve()
                if resolved_image.exists() and resolved_dataset in resolved_image.parents:
                    resolved_image.unlink()
            except Exception as exc:
                LOGGER.warning("Deleted database row but could not remove crop image %s: %s", image_path, exc)

    return True


def delete_face_name(
    custom_name: str,
    *,
    delete_image: bool = True,
    db_path: Path | str = DB_PATH,
    dataset_dir: Path | str = DATASET_DIR,
) -> int:
    """Delete all face entities whose names match case-insensitively."""

    init_database(db_path=db_path, dataset_dir=dataset_dir)
    name_key = _name_key(custom_name)
    if not name_key:
        return 0

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id
            FROM {TABLE_NAME}
            WHERE category = 'Face' AND LOWER(TRIM(custom_name)) = ?
            """,
            (name_key,),
        ).fetchall()

    deleted = 0
    for row in rows:
        if delete_entry(
            int(row["id"]),
            delete_image=delete_image,
            db_path=db_path,
            dataset_dir=dataset_dir,
        ):
            deleted += 1
    return deleted


def delete_detection_event(
    detection_event_id: int,
    *,
    delete_image: bool = True,
    db_path: Path | str = DB_PATH,
    dataset_dir: Path | str = DATASET_DIR,
) -> bool:
    """Delete one stored detection image; delete the entity if it was its master crop."""

    init_database(db_path=db_path, dataset_dir=dataset_dir)
    with _connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT ev.entity_id, ev.image_path, e.image_path AS master_image_path
            FROM {DETECTION_EVENTS_TABLE} ev
            JOIN {TABLE_NAME} e ON e.id = ev.entity_id
            WHERE ev.id = ?
            """,
            (int(detection_event_id),),
        ).fetchone()
        if row is None:
            return False

        event_image_path = Path(str(row["image_path"]))
        master_image_path = Path(str(row["master_image_path"]))
        event_image_path_text = str(row["image_path"])
        master_image_path_text = str(row["master_image_path"])
        entity_id = int(row["entity_id"])

    if event_image_path_text == master_image_path_text:
        return delete_entry(entity_id, delete_image=delete_image, db_path=db_path, dataset_dir=dataset_dir)

    try:
        if event_image_path.expanduser().resolve() == master_image_path.expanduser().resolve():
            return delete_entry(entity_id, delete_image=delete_image, db_path=db_path, dataset_dir=dataset_dir)
    except Exception:
        pass

    with _connect(db_path) as conn:
        conn.execute(
            f"DELETE FROM {DETECTION_EVENTS_TABLE} WHERE id = ?",
            (int(detection_event_id),),
        )
        conn.commit()

    if delete_image:
        try:
            resolved_image = event_image_path.expanduser().resolve()
            resolved_dataset = Path(dataset_dir).expanduser().resolve()
            if resolved_image.exists() and resolved_dataset in resolved_image.parents:
                resolved_image.unlink()
        except Exception as exc:
            LOGGER.warning("Deleted detection event but could not remove crop image %s: %s", event_image_path, exc)

    return True


def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def file_safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_timestamps(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def _prepend_unique_timestamp(existing: Iterable[str], timestamp: str, max_items: int) -> list[str]:
    seen = {timestamp}
    output = [timestamp]
    for item in existing:
        if item not in seen:
            output.append(item)
            seen.add(item)
        if len(output) >= max_items:
            break
    return output[:max_items]


def _update_last_seen(
    conn: sqlite3.Connection,
    row_id: int,
    timestamp: str,
    camera_location: str,
    config: PipelineConfig,
) -> None:
    row = conn.execute(
        f"SELECT last_seen_timestamps FROM {TABLE_NAME} WHERE id = ?",
        (row_id,),
    ).fetchone()
    if row is None:
        return

    updated = _prepend_unique_timestamp(
        _parse_timestamps(row["last_seen_timestamps"]),
        timestamp,
        config.max_timestamps_per_entity,
    )
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET last_seen_timestamps = ?, camera_location = ?
        WHERE id = ?
        """,
        (json.dumps(updated), camera_location, row_id),
    )


def _insert_entry(
    conn: sqlite3.Connection,
    *,
    category: str,
    custom_name: str,
    reg_number: str | None,
    color: str | None,
    vehicle_type: str | None,
    make_model: str | None,
    camera_location: str,
    image_path: str,
    timestamp: str,
    face_embedding: str | None = None,
    image_blob: bytes | None = None,
    image_mime: str | None = None,
) -> int:
    cursor = conn.execute(
        f"""
        INSERT INTO {TABLE_NAME}
            (category, custom_name, reg_number, color, vehicle_type, make_model,
             camera_location, image_path, last_seen_timestamps, image_blob, image_mime, face_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            category,
            custom_name,
            reg_number,
            color,
            vehicle_type,
            make_model,
            camera_location,
            image_path,
            json.dumps([timestamp]),
            image_blob,
            image_mime,
            face_embedding,
        ),
    )
    return int(cursor.lastrowid)


def _encode_crop_jpeg(crop: Any, config: PipelineConfig) -> bytes:
    if cv2 is None:
        raise ImportError("Missing dependency 'opencv-python'. Install it before inference.")
    quality = max(30, min(95, int(config.crop_jpeg_quality)))
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise PipelineRuntimeError("Failed to encode crop image as JPEG.")
    return bytes(encoded)


def _save_crop(crop: Any, prefix: str, timestamp_token: str, config: PipelineConfig) -> StoredCrop:
    if cv2 is None:
        raise ImportError("Missing dependency 'opencv-python'. Install it before inference.")

    filename = f"{prefix}_{timestamp_token}_{uuid.uuid4().hex[:8]}.jpg"
    image_blob = _encode_crop_jpeg(crop, config) if config.store_images_as_blobs else None
    if image_blob is not None and not config.keep_image_files:
        return StoredCrop(image_path=f"db://{filename}", image_blob=image_blob, image_mime="image/jpeg")

    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    image_path = config.dataset_dir / filename
    if image_blob is not None:
        image_path.write_bytes(image_blob)
        ok = True
    else:
        ok = cv2.imwrite(str(image_path), crop)
    if not ok:
        raise PipelineRuntimeError(f"Failed to save crop image: {image_path}")
    return StoredCrop(image_path=str(image_path.resolve()), image_blob=image_blob, image_mime="image/jpeg")


def _normalize_embedding(values: Any) -> Any | None:
    if np is None:
        return None
    try:
        vector = np.asarray(values, dtype="float32").reshape(-1)
    except Exception:
        return None
    if vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _embedding_to_json(embedding: Any | None) -> str | None:
    if embedding is None or np is None:
        return None
    try:
        vector = np.asarray(embedding, dtype="float32").reshape(-1)
    except Exception:
        return None
    return json.dumps([float(value) for value in vector.tolist()])


def _embedding_from_json(raw_value: str | None) -> Any | None:
    if not raw_value or np is None:
        return None
    try:
        values = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return _normalize_embedding(values)


def image_blob_to_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return None


def _decode_image_blob(image_blob: Any) -> Any | None:
    if cv2 is None or np is None:
        return None
    data = image_blob_to_bytes(image_blob)
    if not data:
        return None
    buffer = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _read_crop_from_row(row: sqlite3.Row) -> Any | None:
    crop = _decode_image_blob(row["image_blob"] if "image_blob" in row.keys() else None)
    if crop is not None:
        return crop
    image_path = Path(str(row["image_path"]))
    if image_path.exists() and cv2 is not None:
        return cv2.imread(str(image_path))
    return None


def _cosine_distance(embedding_a: Any, embedding_b: Any) -> float | None:
    if np is None:
        return None
    vector_a = _normalize_embedding(embedding_a)
    vector_b = _normalize_embedding(embedding_b)
    if vector_a is None or vector_b is None:
        return None
    try:
        return 1.0 - float(np.dot(vector_a, vector_b))
    except Exception:
        return None


def _recognition_input_from_crop(face_crop: Any) -> Any | None:
    if cv2 is None or face_crop is None:
        return None
    try:
        height, width = face_crop.shape[:2]
    except Exception:
        return None
    if height < 8 or width < 8:
        return None

    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    squared = cv2.copyMakeBorder(face_crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
    interpolation = cv2.INTER_AREA if side > 112 else cv2.INTER_CUBIC
    return cv2.resize(squared, (112, 112), interpolation=interpolation)


def _extract_insightface_embedding(
    face_crop: Any,
    model_manager: ModelManager,
    config: PipelineConfig,
) -> Any | None:
    if not config.insightface_enabled or cv2 is None or np is None:
        return None

    app = model_manager.insightface_app(config)
    try:
        faces = app.get(face_crop)
    except Exception:
        faces = []

    if faces:
        face = max(
            faces,
            key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
        )
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        normalized = _normalize_embedding(embedding)
        if normalized is not None:
            return normalized

    recognition_input = _recognition_input_from_crop(face_crop)
    if recognition_input is None:
        return None
    recognizer = getattr(app, "models", {}).get("recognition")
    if recognizer is None:
        return None
    try:
        embedding = recognizer.get_feat(recognition_input)
    except Exception as exc:
        LOGGER.debug("InsightFace direct crop embedding failed: %s", exc)
        return None
    return _normalize_embedding(embedding)


def _row_face_embedding(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    model_manager: ModelManager,
    config: PipelineConfig,
) -> Any | None:
    embedding = _embedding_from_json(row["face_embedding"])
    if embedding is not None:
        return embedding

    crop = _read_crop_from_row(row)
    if crop is None:
        return None
    try:
        embedding = _extract_insightface_embedding(crop, model_manager, config)
    except Exception as exc:
        LOGGER.debug("InsightFace embedding backfill failed for row %s: %s", row["id"], exc)
        return None
    embedding_json = _embedding_to_json(embedding)
    if embedding_json:
        conn.execute(
            f"UPDATE {TABLE_NAME} SET face_embedding = ? WHERE id = ?",
            (embedding_json, int(row["id"])),
        )
    return embedding


def sanitize_registration(raw_text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", str(raw_text).upper())
    return cleaned if len(cleaned) >= 3 else UNKNOWN_REGISTRATION


def extract_license_plate_text(
    vehicle_crop: Any,
    model_manager: ModelManager = DEFAULT_MODEL_MANAGER,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Read registration text from the likely plate region of a vehicle crop."""

    config = config or PipelineConfig()
    if vehicle_crop is None:
        return UNKNOWN_REGISTRATION

    try:
        height, width = vehicle_crop.shape[:2]
    except Exception:
        return UNKNOWN_REGISTRATION
    if height < 10 or width < 10:
        return UNKNOWN_REGISTRATION

    y1 = int(height * 0.45)
    y2 = height
    x1 = int(width * 0.05)
    x2 = int(width * 0.95)
    plate_roi = vehicle_crop[y1:y2, x1:x2]

    try:
        _progress(progress_callback, "Initializing OCR reader")
        reader = model_manager.easyocr_reader(config)
        _progress(progress_callback, "Reading license plate text")
        results = reader.readtext(plate_roi, detail=1, paragraph=False)
    except Exception as exc:
        LOGGER.warning("License plate OCR failed; using %s: %s", UNKNOWN_REGISTRATION, exc)
        return UNKNOWN_REGISTRATION

    candidates: list[tuple[str, float]] = []
    for item in results or []:
        try:
            text = str(item[1])
            confidence = float(item[2]) if len(item) > 2 else 0.0
        except Exception:
            continue
        cleaned = sanitize_registration(text)
        if cleaned != UNKNOWN_REGISTRATION:
            candidates.append((cleaned, confidence))

    if candidates:
        candidates.sort(key=lambda candidate: (len(candidate[0]), candidate[1]), reverse=True)
        return candidates[0][0]

    joined = "".join(str(item[1]) for item in results or [] if len(item) > 1)
    return sanitize_registration(joined)


def classify_vehicle_color(vehicle_crop: Any) -> str:
    """Classify paint color from a center-top hood sample in HSV space."""

    if cv2 is None or np is None or vehicle_crop is None:
        return "Unknown"

    try:
        height, width = vehicle_crop.shape[:2]
    except Exception:
        return "Unknown"
    if height < 8 or width < 8:
        return "Unknown"

    x1 = max(0, int(width * 0.35))
    x2 = min(width, int(width * 0.65))
    y1 = max(0, int(height * 0.15))
    y2 = min(height, int(height * 0.35))
    patch = vehicle_crop[y1:y2, x1:x2]
    if getattr(patch, "size", 0) == 0:
        return "Unknown"

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = np.median(hsv.reshape(-1, 3), axis=0)

    if v < 55:
        return "Black"
    if s < 28 and v > 205:
        return "White"
    if s < 45:
        return "Grey/Silver"
    if (h <= 10 or h >= 170) and s >= 60:
        return "Red"
    if 10 < h <= 22 and s >= 60:
        return "Orange/Brown" if v < 165 else "Orange"
    if 22 < h <= 36 and s >= 55:
        return "Yellow/Gold"
    if 36 < h <= 85 and s >= 50:
        return "Green"
    if 85 < h <= 132 and s >= 45:
        return "Blue"
    if 132 < h < 170 and s >= 45:
        return "Purple"
    return "Unknown"


def crop_with_padding(frame: Any, xyxy: Iterable[float], padding: int = 18) -> Any | None:
    """Crop an object with clamped padding around the bounding box."""

    if frame is None:
        return None
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    except Exception:
        return None

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def crop_face_region_from_person_box(frame: Any, xyxy: Iterable[float], padding: int = 18) -> Any | None:
    """Approximate a head/face crop when only a general person detector is available."""

    if frame is None:
        return None
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in xyxy]
    except Exception:
        return None

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    face_x1 = x1 + box_width * 0.18
    face_x2 = x2 - box_width * 0.18
    face_y1 = y1
    face_y2 = y1 + box_height * 0.42

    face_box = (
        max(0, int(round(face_x1)) - padding),
        max(0, int(round(face_y1)) - padding),
        min(width, int(round(face_x2)) + padding),
        min(height, int(round(face_y2)) + padding),
    )
    left, top, right, bottom = face_box
    if right <= left or bottom <= top:
        return crop_with_padding(frame, xyxy, padding)
    return frame[top:bottom, left:right].copy()


def _tensor_to_flat_list(value: Any) -> list[float]:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if np is not None and isinstance(value, np.ndarray):
            return [float(item) for item in value.reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            flattened: list[float] = []
            for item in value:
                flattened.extend(_tensor_to_flat_list(item))
            return flattened
        return [float(value)]
    except Exception:
        return []


def _first_number(value: Any) -> float | None:
    values = _tensor_to_flat_list(value)
    return values[0] if values else None


def _iter_detection_boxes(results: Any) -> Iterable[tuple[list[float], int | None, float | None, int | None]]:
    """Yield ``xyxy, class_id, confidence, track_id`` from Ultralytics output."""

    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            coords = _tensor_to_flat_list(getattr(box, "xyxy", None))
            if len(coords) < 4:
                continue
            class_number = _first_number(getattr(box, "cls", None))
            confidence = _first_number(getattr(box, "conf", None))
            track_number = _first_number(getattr(box, "id", None))
            yield (
                coords[:4],
                int(class_number) if class_number is not None else None,
                float(confidence) if confidence is not None else None,
                int(track_number) if track_number is not None else None,
            )


def _uses_general_person_detector_for_faces(config: PipelineConfig) -> bool:
    """Infer whether the face slot is using a general COCO model instead of a face model."""

    face_path = config.face_model_path
    if face_path is None:
        return False
    if config.vehicle_model_path is not None:
        try:
            if Path(face_path).expanduser().resolve() == Path(config.vehicle_model_path).expanduser().resolve():
                return True
        except Exception:
            pass
    return "face" not in Path(face_path).stem.lower()


def _select_insightface_match(
    candidates: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    candidates.sort(key=lambda item: float(item["distance"]))
    best = candidates[0]
    for candidate in candidates[1:]:
        if _name_key(str(candidate["custom_name"])) == _name_key(str(best["custom_name"])):
            continue
        distance_gap = float(candidate["distance"]) - float(best["distance"])
        if distance_gap < config.insightface_ambiguity_margin:
            LOGGER.info(
                "Ambiguous InsightFace match rejected: best=%s %.4f contender=%s %.4f",
                best["custom_name"],
                best["distance"],
                candidate["custom_name"],
                candidate["distance"],
            )
            return None
    return best


def _find_matching_face_by_insightface(
    conn: sqlite3.Connection,
    face_crop: Any,
    model_manager: ModelManager,
    config: PipelineConfig,
    *,
    exclude_entry_id: int | None = None,
    limit_per_name: bool = True,
) -> dict[str, Any] | None:
    if not config.insightface_enabled:
        return None

    try:
        source_embedding = _extract_insightface_embedding(face_crop, model_manager, config)
    except Exception as exc:
        LOGGER.warning("InsightFace unavailable; leaving face for review: %s", exc)
        return None
    if source_embedding is None:
        LOGGER.debug("InsightFace could not extract an embedding from the source crop.")
        return None

    rows = _fetch_face_match_candidates(
        conn,
        config,
        exclude_entry_id=exclude_entry_id,
        limit_per_name=limit_per_name,
    )
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate_embedding = _row_face_embedding(conn, row, model_manager, config)
        if candidate_embedding is None:
            continue
        distance = _cosine_distance(source_embedding, candidate_embedding)
        if distance is None or distance > config.insightface_match_max_distance:
            continue
        candidates.append(
            {
                "id": int(row["id"]),
                "entity_id": int(row["id"]),
                "custom_name": str(row["custom_name"] or ""),
                "distance": distance,
                "threshold": config.insightface_match_max_distance,
                "matcher": "InsightFace",
            }
        )

    return _select_insightface_match(candidates, config)


def _find_matching_face(
    conn: sqlite3.Connection,
    face_crop: Any,
    model_manager: ModelManager,
    config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> int | None:
    insightface_match = _find_matching_face_by_insightface(conn, face_crop, model_manager, config)
    if insightface_match is not None:
        LOGGER.info(
            "InsightFace matched face to row %s (%s) with distance %.4f",
            insightface_match["id"],
            insightface_match["custom_name"],
            insightface_match["distance"],
        )
        return int(insightface_match["id"])

    _progress(progress_callback, "No confident InsightFace match; leaving face for review")
    return None


def _find_vehicle_by_registration(
    conn: sqlite3.Connection,
    reg_number: str,
    config: PipelineConfig,
) -> int | None:
    if not reg_number:
        return None
    if reg_number == UNKNOWN_REGISTRATION and not config.deduplicate_unknown_registrations:
        return None
    row = conn.execute(
        f"""
        SELECT id FROM {TABLE_NAME}
        WHERE category != 'Face' AND reg_number = ?
        ORDER BY id ASC
        LIMIT 1
        """,
        (reg_number,),
    ).fetchone()
    return int(row["id"]) if row else None


def _update_vehicle_fields_if_empty(
    conn: sqlite3.Connection,
    row_id: int,
    *,
    reg_number: str,
    color: str,
    vehicle_type: str,
) -> None:
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET
            reg_number = CASE
                WHEN (reg_number IS NULL OR reg_number = ? OR reg_number = '') AND ? != ?
                THEN ? ELSE reg_number END,
            color = CASE
                WHEN (color IS NULL OR color = '' OR color = 'Unknown') AND ? != 'Unknown'
                THEN ? ELSE color END,
            vehicle_type = CASE
                WHEN vehicle_type IS NULL OR vehicle_type = ''
                THEN ? ELSE vehicle_type END
        WHERE id = ?
        """,
        (
            UNKNOWN_REGISTRATION,
            reg_number,
            UNKNOWN_REGISTRATION,
            reg_number,
            color,
            color,
            vehicle_type,
            row_id,
        ),
    )


def process_face_crop(
    face_crop: Any,
    camera_location: str,
    *,
    model_manager: ModelManager = DEFAULT_MODEL_MANAGER,
    config: PipelineConfig | None = None,
    allow_identity_matching: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Verify a raw face crop against saved profiles or insert a new entity."""

    config = config or PipelineConfig()
    init_database(config.db_path, config.dataset_dir)
    timestamp = current_timestamp()
    timestamp_token = file_safe_timestamp()

    with _connect(config.db_path) as conn:
        if allow_identity_matching:
            _progress(progress_callback, "Checking face against saved profiles")
            matched_id = _find_matching_face(conn, face_crop, model_manager, config, progress_callback)
            if matched_id is not None:
                _update_last_seen(conn, matched_id, timestamp, camera_location, config)
                detection_crop = _save_crop(face_crop, "FaceSeen", timestamp_token, config)
                _insert_detection_event(
                    conn,
                    entity_id=matched_id,
                    image_path=detection_crop.image_path,
                    image_blob=detection_crop.image_blob,
                    image_mime=detection_crop.image_mime,
                    timestamp=timestamp,
                    camera_location=camera_location,
                )
                conn.commit()
                return {"id": matched_id, "category": "Face", "action": "updated"}
        else:
            _progress(progress_callback, "General person detector in use; sending face to review")

        face_embedding = None
        try:
            face_embedding = _embedding_to_json(
                _extract_insightface_embedding(face_crop, model_manager, config)
            )
        except Exception as exc:
            LOGGER.debug("Could not store InsightFace embedding for new face crop: %s", exc)

        stored_crop = _save_crop(face_crop, "Face", timestamp_token, config)
        row_id = _insert_entry(
            conn,
            category="Face",
            custom_name="Unidentified Face",
            reg_number=None,
            color=None,
            vehicle_type=None,
            make_model=None,
            camera_location=camera_location,
            image_path=stored_crop.image_path,
            timestamp=timestamp,
            face_embedding=face_embedding,
            image_blob=stored_crop.image_blob,
            image_mime=stored_crop.image_mime,
        )
        _insert_detection_event(
            conn,
            entity_id=row_id,
            image_path=stored_crop.image_path,
            image_blob=stored_crop.image_blob,
            image_mime=stored_crop.image_mime,
            timestamp=timestamp,
            camera_location=camera_location,
        )
        conn.commit()
        return {"id": row_id, "category": "Face", "action": "inserted"}


def process_vehicle_crop(
    vehicle_crop: Any,
    vehicle_type: str,
    camera_location: str,
    *,
    track_id: int | None = None,
    runtime_state: DetectionRuntimeState | None = None,
    model_manager: ModelManager = DEFAULT_MODEL_MANAGER,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Extract vehicle intelligence and route insert/update decisions."""

    config = config or PipelineConfig()
    init_database(config.db_path, config.dataset_dir)
    timestamp = current_timestamp()
    timestamp_token = file_safe_timestamp()
    reg_number = extract_license_plate_text(vehicle_crop, model_manager, config, progress_callback)
    _progress(progress_callback, "Classifying vehicle color")
    color = classify_vehicle_color(vehicle_crop)

    with _connect(config.db_path) as conn:
        if runtime_state is not None and track_id is not None:
            row_id = runtime_state.vehicle_track_to_row.get(track_id)
            if row_id is not None:
                _update_last_seen(conn, row_id, timestamp, camera_location, config)
                _update_vehicle_fields_if_empty(
                    conn,
                    row_id,
                    reg_number=reg_number,
                    color=color,
                    vehicle_type=vehicle_type,
                )
                conn.commit()
                return {
                    "id": row_id,
                    "category": vehicle_type,
                    "action": "updated_track",
                    "reg_number": reg_number,
                    "color": color,
                }

        matched_id = _find_vehicle_by_registration(conn, reg_number, config)
        if matched_id is not None:
            _update_last_seen(conn, matched_id, timestamp, camera_location, config)
            _update_vehicle_fields_if_empty(
                conn,
                matched_id,
                reg_number=reg_number,
                color=color,
                vehicle_type=vehicle_type,
            )
            if runtime_state is not None and track_id is not None:
                runtime_state.vehicle_track_to_row[track_id] = matched_id
            conn.commit()
            return {
                "id": matched_id,
                "category": vehicle_type,
                "action": "updated_registration",
                "reg_number": reg_number,
                "color": color,
            }

        stored_crop = _save_crop(vehicle_crop, "Vehicle", timestamp_token, config)
        row_id = _insert_entry(
            conn,
            category=vehicle_type,
            custom_name=f"Vehicle_{timestamp_token}",
            reg_number=reg_number,
            color=color,
            vehicle_type=vehicle_type,
            make_model="",
            camera_location=camera_location,
            image_path=stored_crop.image_path,
            timestamp=timestamp,
            image_blob=stored_crop.image_blob,
            image_mime=stored_crop.image_mime,
        )
        if runtime_state is not None and track_id is not None:
            runtime_state.vehicle_track_to_row[track_id] = row_id
        conn.commit()
        return {
            "id": row_id,
            "category": vehicle_type,
            "action": "inserted",
            "reg_number": reg_number,
            "color": color,
        }


def process_frame(
    frame: Any,
    camera_location: str,
    *,
    model_manager: ModelManager = DEFAULT_MODEL_MANAGER,
    config: PipelineConfig | None = None,
    runtime_state: DetectionRuntimeState | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Detect faces and vehicles in one frame and persist entity records."""

    if cv2 is None or np is None:
        raise ImportError("Missing runtime dependencies 'opencv-python' and/or 'numpy'.")

    config = config or PipelineConfig()
    runtime_state = runtime_state or DetectionRuntimeState()
    detections: list[dict[str, Any]] = []

    _progress(progress_callback, "Loading vehicle YOLO model")
    vehicle_model = model_manager.vehicle_model(config)
    try:
        _progress(progress_callback, "Running vehicle detection")
        vehicle_results = vehicle_model.track(
            frame,
            persist=True,
            classes=list(COCO_VEHICLE_CLASSES.keys()),
            conf=config.vehicle_confidence,
            verbose=False,
        )
    except Exception as exc:
        LOGGER.warning("YOLO vehicle tracking failed; falling back to prediction: %s", exc)
        _progress(progress_callback, "Vehicle tracking failed; running vehicle prediction")
        vehicle_results = vehicle_model.predict(
            frame,
            classes=list(COCO_VEHICLE_CLASSES.keys()),
            conf=config.vehicle_confidence,
            verbose=False,
        )

    for xyxy, class_id, confidence, track_id in _iter_detection_boxes(vehicle_results):
        if class_id not in COCO_VEHICLE_CLASSES:
            continue
        vehicle_type = COCO_VEHICLE_CLASSES[class_id]
        crop = crop_with_padding(frame, xyxy, config.crop_padding_pixels)
        if crop is None:
            continue
        try:
            _progress(progress_callback, f"Processing {vehicle_type} crop")
            record = process_vehicle_crop(
                crop,
                vehicle_type,
                camera_location,
                track_id=track_id,
                runtime_state=runtime_state,
                model_manager=model_manager,
                config=config,
                progress_callback=progress_callback,
            )
            record["confidence"] = confidence
            record["track_id"] = track_id
            detections.append(record)
        except Exception as exc:
            LOGGER.exception("Vehicle crop processing failed: %s", exc)
            detections.append({"category": vehicle_type, "action": "error", "error": str(exc)})

    _progress(progress_callback, "Loading face YOLO model")
    face_model = model_manager.face_model(config)
    use_person_boxes = _uses_general_person_detector_for_faces(config)
    if use_person_boxes:
        _progress(progress_callback, "Running people detection with general YOLO model; auto face matching disabled")
        face_results = face_model.predict(
            frame,
            classes=[COCO_PERSON_CLASS_ID],
            conf=config.face_confidence,
            verbose=False,
        )
    else:
        _progress(progress_callback, "Running face detection")
        face_results = face_model.predict(frame, conf=config.face_confidence, verbose=False)
    for xyxy, _class_id, confidence, _track_id in _iter_detection_boxes(face_results):
        if use_person_boxes:
            crop = crop_face_region_from_person_box(frame, xyxy, config.crop_padding_pixels)
        else:
            crop = crop_with_padding(frame, xyxy, config.crop_padding_pixels)
        if crop is None:
            continue
        try:
            _progress(progress_callback, "Processing face crop")
            record = process_face_crop(
                crop,
                camera_location,
                model_manager=model_manager,
                config=config,
                allow_identity_matching=not use_person_boxes,
                progress_callback=progress_callback,
            )
            record["confidence"] = confidence
            detections.append(record)
        except Exception as exc:
            LOGGER.exception("Face crop processing failed: %s", exc)
            detections.append({"category": "Face", "action": "error", "error": str(exc)})

    return detections


def infer_source_type(source: str | Path, source_type: str | None = None) -> str:
    """Infer canonical source type: ``live``, ``video``, or ``image``."""

    if source_type:
        normalized = source_type.strip().lower().replace("_", " ")
        if normalized in {"live", "live stream", "rtsp", "live eufy stream"}:
            return "live"
        if normalized in {"video", "video file", "upload video file"}:
            return "video"
        if normalized in {"image", "image file", "upload image file"}:
            return "image"
        raise ValueError(f"Unsupported source_type: {source_type}")

    source_text = str(source).strip()
    lower_source = source_text.lower()
    if lower_source.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return "live"
    suffix = Path(source_text).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Unable to infer source type. Provide source_type='live', 'video', or 'image'.")


def _open_capture(source: str | Path) -> Any:
    if cv2 is None:
        raise ImportError("Missing dependency 'opencv-python'.")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise PipelineRuntimeError(f"Unable to open video source: {source}")
    return capture


def run_pipeline(
    source: str | Path,
    *,
    source_type: str | None = None,
    camera_location: str | None = None,
    stop_event: threading.Event | None = None,
    model_manager: ModelManager = DEFAULT_MODEL_MANAGER,
    config: PipelineConfig | None = None,
    frame_callback: Callable[[Any], None] | None = None,
    result_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    max_processed_frames: int | None = None,
) -> dict[str, Any]:
    """Run the pipeline against an RTSP stream, video file, or image file."""

    config = config or PipelineConfig()
    _progress(progress_callback, "Initializing database")
    init_database(config.db_path, config.dataset_dir)
    stop_event = stop_event or threading.Event()

    source_kind = infer_source_type(source, source_type)
    location = (camera_location or "").strip() or Path(str(source)).name or str(source)
    runtime_state = DetectionRuntimeState()

    summary: dict[str, Any] = {
        "source": str(source),
        "source_type": source_kind,
        "camera_location": location,
        "processed_frames": 0,
        "detections": 0,
        "status": "started",
    }

    if source_kind == "image":
        if cv2 is None:
            raise ImportError("Missing dependency 'opencv-python'.")
        _progress(progress_callback, "Reading uploaded image")
        frame = cv2.imread(str(source))
        if frame is None:
            raise PipelineRuntimeError(f"Failed to read image file: {source}")
        _progress(progress_callback, "Processing image frame")
        detections = process_frame(
            frame,
            location,
            model_manager=model_manager,
            config=config,
            runtime_state=runtime_state,
            progress_callback=progress_callback,
        )
        if result_callback:
            result_callback(detections)
        if frame_callback:
            frame_callback(frame)
        summary.update(processed_frames=1, detections=len(detections), status="completed")
        _progress(progress_callback, f"Completed image processing with {len(detections)} detections")
        return summary

    skip_frames = config.live_skip_frames if source_kind == "live" else config.local_skip_frames
    process_interval = max(1, skip_frames + 1)
    _progress(progress_callback, "Opening video source")
    capture = _open_capture(source)
    reconnect_attempts = 0
    frame_index = 0

    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                if source_kind == "live" and reconnect_attempts < config.max_reconnect_attempts:
                    reconnect_attempts += 1
                    LOGGER.warning(
                        "Live stream dropped. Reconnect attempt %s/%s in %.1fs",
                        reconnect_attempts,
                        config.max_reconnect_attempts,
                        config.reconnect_delay_seconds,
                    )
                    capture.release()
                    _progress(progress_callback, f"Live stream dropped; reconnecting ({reconnect_attempts}/{config.max_reconnect_attempts})")
                    time.sleep(config.reconnect_delay_seconds)
                    capture = _open_capture(source)
                    continue
                summary["status"] = "stream_ended_or_dropped"
                break

            reconnect_attempts = 0
            frame_index += 1
            if (frame_index - 1) % process_interval != 0:
                continue

            _progress(progress_callback, f"Processing frame {frame_index}")
            detections = process_frame(
                frame,
                location,
                model_manager=model_manager,
                config=config,
                runtime_state=runtime_state,
                progress_callback=progress_callback,
            )
            summary["processed_frames"] += 1
            summary["detections"] += len(detections)

            if result_callback:
                result_callback(detections)
            if frame_callback:
                frame_callback(frame)

            if max_processed_frames is not None and summary["processed_frames"] >= max_processed_frames:
                summary["status"] = "max_processed_frames_reached"
                break

        if stop_event.is_set():
            summary["status"] = "stopped"
        elif summary.get("status") == "started":
            summary["status"] = "completed"
    finally:
        capture.release()

    return summary


__all__ = [
    "COCO_VEHICLE_CLASSES",
    "DATASET_DIR",
    "DB_PATH",
    "DetectionRuntimeState",
    "EASYOCR_GPU_ENV",
    "FACE_MODEL_ENV",
    "ModelManager",
    "PipelineConfig",
    "PipelineRuntimeError",
    "TABLE_NAME",
    "UNKNOWN_REGISTRATION",
    "VEHICLE_MODEL_ENV",
    "classify_vehicle_color",
    "crop_face_region_from_person_box",
    "crop_with_padding",
    "delete_entry",
    "delete_face_name",
    "delete_detection_event",
    "discover_default_model_path",
    "extract_license_plate_text",
    "fetch_face_detections_for_name",
    "fetch_entries",
    "fetch_named_faces",
    "infer_source_type",
    "init_database",
    "image_blob_to_bytes",
    "process_face_crop",
    "process_frame",
    "process_vehicle_crop",
    "run_pipeline",
    "sanitize_registration",
    "suggest_named_face_for_entry",
    "update_entry",
]
