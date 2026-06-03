# Copyright 2026 Sentry Object Intelligence contributors
# SPDX-License-Identifier: Apache-2.0

"""Streamlit dashboard for the face and vehicle intelligence pipeline."""

from __future__ import annotations

import html
import json
import queue
import re
import tempfile
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

import pipeline


APP_TITLE = "Sentry Object Intelligence"
UPLOAD_CACHE_DIR = Path(tempfile.gettempdir()) / "sentry_uploads"


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name).strip("._")
    return cleaned or "uploaded_source"


def _save_uploaded_file(uploaded_file: Any) -> Path:
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = UPLOAD_CACHE_DIR / f"{timestamp}_{_safe_filename(uploaded_file.name)}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def _parse_timestamps(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed[:6]]


def _is_placeholder_name(name: Any) -> bool:
    return bool(re.fullmatch(r"(Face|Vehicle)_\d{8}_\d{6}", str(name or "").strip()))


def _display_name(entry: dict[str, Any], *, is_vehicle: bool) -> str:
    raw_name = str(entry.get("custom_name") or "").strip()
    if raw_name and not _is_placeholder_name(raw_name):
        return raw_name
    return "Unidentified Vehicle" if is_vehicle else "Unidentified Face"


def _editable_name(entry: dict[str, Any]) -> str:
    raw_name = str(entry.get("custom_name") or "").strip()
    if raw_name == "Unidentified Face":
        return ""
    return "" if _is_placeholder_name(raw_name) else raw_name


def _is_unidentified_face(entry: dict[str, Any]) -> bool:
    raw_name = str(entry.get("custom_name") or "").strip()
    return not raw_name or raw_name == "Unidentified Face" or _is_placeholder_name(raw_name)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - older Streamlit
        st.experimental_rerun()


def _ensure_session_state() -> None:
    st.session_state.setdefault("processor_thread", None)
    st.session_state.setdefault("processor_stop_event", None)
    st.session_state.setdefault("status_queue", queue.Queue())
    st.session_state.setdefault("last_worker_message", "Idle")


def _is_processing() -> bool:
    worker = st.session_state.get("processor_thread")
    return bool(worker and worker.is_alive())


def _pipeline_worker(
    source: str,
    source_type: str,
    camera_location: str,
    config: pipeline.PipelineConfig,
    stop_event: threading.Event,
    status_queue: "queue.Queue[tuple[str, str]]",
) -> None:
    try:
        status_queue.put(("progress", "Worker started"))
        summary = pipeline.run_pipeline(
            source,
            source_type=source_type,
            camera_location=camera_location,
            config=config,
            stop_event=stop_event,
            progress_callback=lambda message: status_queue.put(("progress", message)),
        )
        status_queue.put(("success", f"{summary['status']} | frames={summary['processed_frames']} detections={summary['detections']}"))
    except Exception as exc:  # pragma: no cover - runtime/model specific
        status_queue.put(("error", str(exc)))


def _drain_status_queue() -> None:
    status_queue = st.session_state.status_queue
    while True:
        try:
            level, message = status_queue.get_nowait()
        except queue.Empty:
            break
        st.session_state.last_worker_message = message
        if level == "error":
            st.toast(f"Pipeline error: {message}")
        elif level == "success":
            st.toast("Processing complete")
            st.session_state["refresh_after_worker"] = True
        elif level == "progress":
            pass


def _build_pipeline_config(
    vehicle_model_path: str,
    face_model_path: str,
    *,
    show_errors: bool = False,
) -> pipeline.PipelineConfig | None:
    base_config = pipeline.PipelineConfig.with_model_paths(
        vehicle_model_path=vehicle_model_path.strip() or None,
        face_model_path=face_model_path.strip() or None,
    )
    if base_config.vehicle_model_path is None:
        if show_errors:
            st.sidebar.error("No vehicle model path provided and no .pt file found in models/.")
        return None
    if base_config.face_model_path is None:
        if show_errors:
            st.sidebar.error("No face model path provided and no .pt file found in models/.")
        return None

    return replace(
        base_config,
        db_path=pipeline.DB_PATH,
        dataset_dir=pipeline.DATASET_DIR,
    )


def _start_or_stop_processing(
    selected_source_type: str,
    camera_location: str,
    rtsp_url: str,
    uploaded_file: Any,
    config: pipeline.PipelineConfig | None,
) -> None:
    if _is_processing():
        stop_event = st.session_state.get("processor_stop_event")
        if stop_event is not None:
            stop_event.set()
        st.session_state.last_worker_message = "Stop requested. Finishing the current frame..."
        return

    if config is None:
        return

    try:
        if selected_source_type == "Live Eufy Stream":
            if not rtsp_url.strip():
                st.sidebar.error("RTSP URL is required.")
                return
            source = rtsp_url.strip()
            canonical_type = "live"
        elif selected_source_type == "Upload Video File":
            if uploaded_file is None:
                st.sidebar.error("Choose a video file first.")
                return
            source = str(_save_uploaded_file(uploaded_file))
            canonical_type = "video"
        else:
            if uploaded_file is None:
                st.sidebar.error("Choose an image file first.")
                return
            source = str(_save_uploaded_file(uploaded_file))
            canonical_type = "image"
    except Exception as exc:
        st.sidebar.error(f"Upload failed: {exc}")
        return

    location = camera_location.strip() or selected_source_type
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_pipeline_worker,
        args=(source, canonical_type, location, config, stop_event, st.session_state.status_queue),
        daemon=True,
    )
    st.session_state.processor_stop_event = stop_event
    st.session_state.processor_thread = worker
    st.session_state.last_worker_message = f"Processing {selected_source_type}"
    worker.start()


def _render_sidebar() -> None:
    st.sidebar.header("Input Controller")
    selected_source_type = st.sidebar.selectbox(
        "Source type",
        ["Live Eufy Stream", "Upload Video File", "Upload Image File"],
    )
    camera_location = st.sidebar.text_input("Camera Location", value="Front Door")

    rtsp_url = ""
    uploaded_file = None
    if selected_source_type == "Live Eufy Stream":
        rtsp_url = st.sidebar.text_input("RTSP URL", type="password")
    elif selected_source_type == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Video file",
            type=["mp4", "avi", "mov", "mkv", "m4v", "webm"],
            accept_multiple_files=False,
        )
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Image file",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
            accept_multiple_files=False,
        )

    default_config = pipeline.PipelineConfig()
    detected_vehicle_model_path = str(default_config.vehicle_model_path or "")
    detected_face_model_path = str(default_config.face_model_path or "")
    st.session_state.setdefault("vehicle_model_path_input", detected_vehicle_model_path)
    st.session_state.setdefault("face_model_path_input", detected_face_model_path)

    with st.sidebar.expander("Model Paths", expanded=True):
        if st.button("Use Detected Models", use_container_width=True):
            st.session_state.vehicle_model_path_input = detected_vehicle_model_path
            st.session_state.face_model_path_input = detected_face_model_path
            _rerun()

        vehicle_model_path = st.text_input(
            "Vehicle YOLO model",
            key="vehicle_model_path_input",
        )
        face_model_path = st.text_input(
            "Face YOLO model",
            key="face_model_path_input",
        )
        st.caption("Face recognition uses InsightFace buffalo_l embeddings when available.")

    running = _is_processing()
    button_label = "Stop Processing" if running else "Start Processing"
    button_type = "secondary" if running else "primary"
    if st.sidebar.button(button_label, type=button_type, use_container_width=True):
        config = None
        if not running:
            config = _build_pipeline_config(
                vehicle_model_path,
                face_model_path,
                show_errors=True,
            )
        _start_or_stop_processing(
            selected_source_type,
            camera_location,
            rtsp_url,
            uploaded_file,
            config,
        )

    st.sidebar.divider()
    if _is_processing():
        st.sidebar.success("Processing thread running")
        st.sidebar.progress(0, text=st.session_state.last_worker_message)
    else:
        st.sidebar.info("Processing thread idle")
    st.sidebar.caption(st.session_state.last_worker_message)


def _image_card(image_path: str, image_blob: Any = None) -> None:
    image_bytes = pipeline.image_blob_to_bytes(image_blob)
    if image_bytes:
        try:
            st.image(image_bytes, use_container_width=True)
        except TypeError:  # pragma: no cover - older Streamlit
            st.image(image_bytes, use_column_width=True)
        return

    path = Path(image_path) if image_path else None
    if path and path.exists():
        try:
            st.image(str(path), use_container_width=True)
        except TypeError:  # pragma: no cover - older Streamlit
            st.image(str(path), use_column_width=True)
    else:
        st.warning("Image not found")


def _card_delete_button(key: str) -> bool:
    action_col, label_col = st.columns([1, 8])
    with action_col:
        clicked = st.button("x", key=key, help="Delete this card")
    with label_col:
        st.caption("Delete")
    return clicked


def _render_detection_timeline(raw_timestamps: str | None) -> None:
    timestamps = _parse_timestamps(raw_timestamps)
    previous_timestamps = timestamps[1:6]
    st.markdown("**Previous 5 detections**")
    if previous_timestamps:
        st.markdown(
            "\n".join(f"- {html.escape(timestamp)}" for timestamp in previous_timestamps),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No previous detections recorded")


def _render_entry_card(entry: dict[str, Any], *, is_vehicle: bool) -> None:
    suggested_name = None
    is_review_face = not is_vehicle and _is_unidentified_face(entry)
    if is_review_face:
        suggestion_key = f"suggested_match_{entry['id']}"
        suggested_name = st.session_state.get(suggestion_key)
        if _card_delete_button(f"review_delete_x_{entry['id']}"):
            pipeline.delete_entry(int(entry["id"]))
            st.session_state["last_worker_message"] = "Detection deleted"
            _rerun()

    title = html.escape(suggested_name or _display_name(entry, is_vehicle=is_vehicle))
    st.markdown(f"### {title}", unsafe_allow_html=True)
    if suggested_name:
        st.caption("Suggested existing match")
    _image_card(str(entry.get("image_path") or ""), entry.get("image_blob"))

    if is_vehicle:
        reg_number = html.escape(str(entry.get("reg_number") or pipeline.UNKNOWN_REGISTRATION))
        vehicle_type = html.escape(str(entry.get("vehicle_type") or entry.get("category") or "Vehicle"))
        color = html.escape(str(entry.get("color") or "Unknown"))
        make_model = html.escape(str(entry.get("make_model") or ""))
        location = html.escape(str(entry.get("camera_location") or "Unknown"))
        st.markdown(
            "\n".join(
                [
                    f"**Reg:** `{reg_number}`",
                    f"**Type:** {vehicle_type}",
                    f"**Color:** {color}",
                    f"**Make / Model:** {make_model or 'Not set'}",
                    f"**Last location:** {location}",
                ]
            ),
            unsafe_allow_html=True,
        )
    else:
        location = html.escape(str(entry.get("camera_location") or "Unknown"))
        st.markdown(f"**Last location:** {location}", unsafe_allow_html=True)

    _render_detection_timeline(entry.get("last_seen_timestamps"))

    if is_review_face:
        if st.button("Check Existing Match", key=f"suggest_match_{entry['id']}", use_container_width=True):
            with st.spinner("Checking saved faces..."):
                try:
                    suggestion = pipeline.suggest_named_face_for_entry(int(entry["id"]))
                except Exception as exc:
                    st.warning(f"Could not check match: {exc}")
                    suggestion = None
            if suggestion:
                st.session_state[f"suggested_match_{entry['id']}"] = str(suggestion.get("custom_name") or "")
            else:
                st.session_state.pop(f"suggested_match_{entry['id']}", None)
                st.info("No confident existing match found.")
            _rerun()

    with st.form(key=f"edit_{entry['id']}", clear_on_submit=False):
        suggested_name = st.session_state.get(f"suggested_match_{entry['id']}") if not is_vehicle else None
        new_name = st.text_input("Display name", value=suggested_name or _editable_name(entry))
        new_make_model = None
        if is_vehicle:
            new_make_model = st.text_input("Vehicle make and model", value=str(entry.get("make_model") or ""))
        submitted = st.form_submit_button("Save Changes", use_container_width=True)
        if submitted:
            if not new_name.strip():
                st.warning("Enter a display name before saving.")
                return
            pipeline.update_entry(
                int(entry["id"]),
                custom_name=new_name,
                make_model=new_make_model if is_vehicle else None,
            )
            st.session_state["last_worker_message"] = f"Saved {_display_name({'custom_name': new_name}, is_vehicle=is_vehicle)}"
            _rerun()


def _render_grid(entries: list[dict[str, Any]], *, is_vehicle: bool) -> None:
    if not entries:
        st.info("No records logged yet.")
        return

    for start in range(0, len(entries), 3):
        columns = st.columns(3)
        for column, entry in zip(columns, entries[start : start + 3]):
            with column:
                try:
                    card = st.container(border=True)
                except TypeError:  # pragma: no cover - older Streamlit
                    card = st.container()
                with card:
                    _render_entry_card(entry, is_vehicle=is_vehicle)


def _render_people_gallery() -> None:
    people = pipeline.fetch_named_faces()
    if not people:
        st.info("No named faces yet. Name detections in the review queue and they will move here.")
        return

    search_query = st.text_input("Search by name", placeholder="Regex, case-sensitive")
    regex_query = search_query.strip()
    if regex_query:
        try:
            name_pattern = re.compile(regex_query)
        except re.error as exc:
            st.warning(f"Invalid regex: {exc}")
            filtered_people = []
        else:
            filtered_people = [
                person
                for person in people
                if name_pattern.search(str(person.get("custom_name") or ""))
            ]
    else:
        filtered_people = people
    visible_people = filtered_people[:20]
    if not visible_people:
        st.info("No names match that search.")
        return

    table_rows = []
    for person in visible_people:
        timestamps = person.get("last_detection_timestamps") or []
        table_rows.append(
            {
                "Name": str(person.get("custom_name") or ""),
                "Detections": int(person.get("detection_count") or 0),
                "Last 2 detection timestamps": " | ".join(str(item) for item in timestamps[:2]) or "None",
            }
        )

    selected_index = 0
    try:
        selection = st.dataframe(
            table_rows,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_rows = list(getattr(selection.selection, "rows", []) or [])
        if selected_rows:
            selected_index = int(selected_rows[0])
    except TypeError:  # pragma: no cover - older Streamlit
        st.dataframe(table_rows, hide_index=True, use_container_width=True)
        selected_name_fallback = st.selectbox(
            "Open detected name",
            [str(person.get("custom_name") or "") for person in visible_people],
        )
        selected_index = next(
            (
                index
                for index, person in enumerate(visible_people)
                if str(person.get("custom_name") or "") == selected_name_fallback
            ),
            0,
        )

    selected_index = min(selected_index, len(visible_people) - 1)
    selected_name = str(visible_people[selected_index]["custom_name"])
    st.divider()
    st.subheader(selected_name)

    detections = pipeline.fetch_face_detections_for_name(selected_name)
    if not detections:
        st.info("No detection images stored for this person.")
        return

    timestamps: list[str] = []
    for detection in detections:
        detected_at = str(detection.get("detected_at") or "")
        if detected_at and detected_at not in timestamps:
            timestamps.append(detected_at)
    if timestamps:
        st.markdown("**Detection timeline**")
        st.markdown("\n".join(f"- {html.escape(timestamp)}" for timestamp in timestamps[:5]), unsafe_allow_html=True)

    for start in range(0, len(detections), 4):
        columns = st.columns(4)
        for column, detection in zip(columns, detections[start : start + 4]):
            with column:
                try:
                    card = st.container(border=True)
                except TypeError:  # pragma: no cover - older Streamlit
                    card = st.container()
                with card:
                    event_id = int(detection["detection_event_id"])
                    if _card_delete_button(f"detection_delete_x_{event_id}"):
                        pipeline.delete_detection_event(event_id)
                        st.session_state["last_worker_message"] = "Detection image deleted"
                        _rerun()
                    _image_card(str(detection.get("image_path") or ""), detection.get("image_blob"))
                    st.caption(str(detection.get("detected_at") or "Unknown time"))
                    location = str(detection.get("camera_location") or "Unknown location")
                    st.caption(location)


def _render_dashboard() -> None:
    entries = pipeline.fetch_entries()
    face_review_queue = [
        entry
        for entry in entries
        if entry.get("category") == "Face" and _is_unidentified_face(entry)
    ]
    vehicles = [entry for entry in entries if entry.get("category") != "Face"]

    controls, db_label = st.columns([1, 5])
    with controls:
        if st.button("Refresh", use_container_width=True):
            _rerun()
    with db_label:
        st.caption(f"SQLite database: `{pipeline.DB_PATH}`")

    review_tab, people_tab, vehicle_tab = st.tabs(["👤 Face Review", "🪪 Detected Names", "🚘 Logged Vehicles"])
    with review_tab:
        st.subheader(f"Face Review Queue ({len(face_review_queue)})")
        st.caption("Named faces move out of this queue and into Detected Names.")
        _render_grid(face_review_queue, is_vehicle=False)
    with people_tab:
        _render_people_gallery()
    with vehicle_tab:
        st.subheader(f"Logged Vehicles ({len(vehicles)})")
        _render_grid(vehicles, is_vehicle=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    pipeline.init_database()
    _ensure_session_state()
    _drain_status_queue()
    if st.session_state.pop("refresh_after_worker", False):
        _rerun()

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {background: #f4f7f6;}
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #d9e2df;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
        .stTabs [data-baseweb="tab"] {padding: 0.45rem 0.9rem;}
        img {border-radius: 6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _render_sidebar()
    st.title(APP_TITLE)
    _render_dashboard()

    if _is_processing():
        time.sleep(1.5)
        _rerun()


if __name__ == "__main__":
    main()
