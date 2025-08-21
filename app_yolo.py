# app_yolo.py — Streamlit app using Ultralytics YOLO + ByteTrack
# - Detects fish (expects a model trained with a "goldfish" class, or generic "fish")
# - Tracks with ByteTrack (persist=True)
# - Applies stillness + top/bottom dwell logic per tracked goldfish
# Run:
#   pip install -r requirements.txt
#   streamlit run app_yolo.py

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

try:
    import cv2
except Exception as e:
    import streamlit as st, platform, sys, numpy
    st.error(
        "OpenCV failed to import. Most common causes: wrong wheel, missing libGL, "
        "or incompatible NumPy. Details below."
    )
    st.code(
        f"Python={sys.version}\n"
        f"Platform={platform.platform()}\n"
        f"NumPy import OK: {hasattr(numpy, '__version__')} (numpy {getattr(numpy,'__version__','?')})\n"
        f"Error: {repr(e)}"
    )
    st.stop()

import time
from typing import List
import numpy as np
import streamlit as st

from yolo_infer import YOLOTracker
from track_store import TrackStore
from decision import compute_avg_speed, is_likely_dead
from classic_detector import ClassicDetector

st.set_page_config(page_title="Dead Goldfish Detector (YOLO)", layout="wide")
st.title("🐟 Dead Goldfish Detector — YOLO + ByteTrack")

st.markdown(
    """
This scaffold uses **Ultralytics YOLO** (detection) + **ByteTrack** (tracking) and then applies a **simple rule** to flag
**likely dead** goldfish:
- Nearly **motionless** for *N* seconds, **and**
- Dwelling at the **top** (floating) or **bottom** (sunk) zone of the tank.

👉 Train your model with a `goldfish` class (and optionally `other_fish`, `debris`) and point the app to your `.pt`.
If your model is single-class `fish`, enable the color gate below to mark orange fish as goldfish.
"""
)

# ---------------------
# Sidebar controls
# ---------------------
with st.sidebar:
    st.header("Model")

    default_weights = (
        "weights/goldfish_best.pt"
        if os.path.exists("weights/goldfish_best.pt")
        else "yolov8n.pt"
    )
    model_path = st.text_input(
        "YOLO model path (.pt)",
        value=default_weights,
        help="Path or URL to trained weights (e.g., weights/goldfish_best.pt)",
    )
    conf = st.slider("Confidence threshold", 0.05, 0.8, 0.25, 0.01)
    iou = st.slider("NMS IoU", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.selectbox("Image size", [480, 512, 640, 960, 1280], index=2)
    tracker_alg = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)
    device = st.selectbox(
        "Device", ["auto", "cpu"], index=0, help="Use 'auto' for GPU if available, else CPU"
    )

    st.header("Classes")
    st.write("Which detected classes should be treated as **goldfish** for dead/alive logic?")
    goldfish_class_keywords = st.text_input(
        "Goldfish class keywords (comma-separated)", value="goldfish"
    )
    ignore_other_classes = st.checkbox("Ignore non-goldfish classes", value=True)
    use_orange_gate = st.checkbox(
        "If class == 'fish', mark orange fish as goldfish", value=True,
        help="Useful if you trained a single-class model named 'fish'"
    )

    st.header("Video Source")
    source_mode = st.radio("Choose source", ["Webcam (0)", "RTSP URL", "Upload MP4"], index=0)
    rtsp_url = st.text_input("RTSP URL", value="", help="e.g., rtsp://user:pass@ip/stream")
    uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mkv"])

    st.header("Decision Rules")
    window_sec = st.slider("Stillness window (seconds)", 5, 180, 30)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 300, 25)
    top_zone_pct = st.slider("Top zone height (%)", 2, 40, 12)
    bottom_zone_pct = st.slider("Bottom zone height (%)", 2, 40, 12)
    use_fallback = st.checkbox("Fallback to classic detector if YOLO finds 0", value=True)

    st.header("Run")
    start = st.button("▶️ Start")
    stop = st.button("⏹ Stop")
    if "running" not in st.session_state:
        st.session_state.running = False
    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

# Auto-switch to Upload MP4 if a file is provided
if uploaded_file and source_mode != "Upload MP4":
    source_mode = "Upload MP4"

# ---------------------
# Helpers
# ---------------------
def open_capture(mode: str, rtsp_url: str, uploaded_file):
    if mode == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    elif mode == "RTSP URL":
        cap = cv2.VideoCapture(rtsp_url)
    else:
        if uploaded_file is None:
            return None
        # Preserve extension so FFMPEG picks the right demuxer
        ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tmp_path = f"tmp_upload{ext}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp_path)  # consider cv2.CAP_FFMPEG depending on build
    if not cap or not cap.isOpened():
        return None
    return cap

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def class_is_goldfish(name: str, keywords: List[str]) -> bool:
    n = (name or "").lower()
    return any(kw.strip().lower() in n for kw in keywords if kw.strip())

def orange_ratio(bgr_roi) -> float:
    # Rough orange detector in HSV (tune if needed)
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    lower = np.array((5, 80, 80), np.uint8)
    upper = np.array((30, 255, 255), np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / float(mask.size)

def assign_pseudo_ids(dets, store, max_dist=60.0):
    """
    Ensure every det has a tid.
    For tid=None, assign to nearest existing track (last point) within max_dist,
    else create a new pseudo id using a counter in session_state.
    """
    if "pseudo_tid" not in st.session_state:
        st.session_state.pseudo_tid = -1

    # Build list of (tid, last_x, last_y)
    last_points = []
    for tid, tr in store.tracks.items():
        if tr.history:
            last_points.append((tid, tr.history[-1][1], tr.history[-1][2]))

    for d in dets:
        if d.get("tid") is not None:
            continue
        cx, cy = d["center"]
        # nearest neighbor to existing tracks
        best_tid, best_dist2 = None, float("inf")
        for tid, lx, ly in last_points:
            dx, dy = cx - lx, cy - ly
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_tid = tid
        if best_tid is not None and best_dist2 <= (max_dist * max_dist):
            d["tid"] = best_tid
        else:
            st.session_state.pseudo_tid -= 1  # negative IDs for pseudo
            d["tid"] = st.session_state.pseudo_tid

# ---------------------
# Init
# ---------------------
alerts_dir = "alerts"
ensure_dir(alerts_dir)

frame_area_text = st.empty()
preview = st.empty()
alerts_panel = st.container()
metrics_ph = st.container()

cap = open_capture(source_mode, rtsp_url, uploaded_file)
if not cap:
    st.warning("Provide a valid video source (webcam 0, RTSP URL, or upload a video).")
    st.stop()

# Load YOLO (fallback to yolov8n.pt if local path is missing and not a URL)
if (
    model_path
    and model_path.endswith(".pt")
    and not (model_path.startswith("http://") or model_path.startswith("https://"))
    and not os.path.exists(model_path)
):
    st.warning(
        f"Model '{model_path}' not found locally. Falling back to 'yolov8n.pt' for a smoke test."
    )
    model_path = "yolov8n.pt"

try:
    yolo = YOLOTracker(
        model_path=model_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        tracker=tracker_alg,
        device=None if device == "auto" else device,
    )
    # Show class list once for sanity
    model_names = list(getattr(yolo.model, "names", {}).values())
    st.sidebar.caption(f"Model classes: {model_names}")
except Exception as e:
    st.error(f"Failed to load YOLO model from '{model_path}': {e}")
    st.stop()

store = TrackStore(maxlen=6000, stale_sec=12)
classic = ClassicDetector()  # for optional fallback

last_frame_t = time.time()
fps_smooth = 0.0
keywords = [k for k in goldfish_class_keywords.split(",")]

while st.session_state.running:
    ok, frame = cap.read()
    if not ok:
        st.info("End of stream or cannot read frames.")
        break

    now_t = time.time()
    dt = now_t - last_frame_t
    last_frame_t = now_t
    fps = 1.0 / dt if dt > 1e-3 else 0.0
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

    h, w = frame.shape[:2]
    frame_area_text.write(f"Frame: {w}×{h} | FPS: {fps_smooth:.1f}")

    # YOLO detect+track in one call
    dets = yolo.infer(frame)

    # Optional fallback if YOLO returns nothing
    if use_fallback and len(dets) == 0:
        dets = classic.infer(frame)

    # Ensure every det has a tid so TrackStore can track motion
    assign_pseudo_ids(dets, store, max_dist=60.0)

    # Optionally filter to only "goldfish" classes for logic
    filtered = []
    for d in dets:
        cls_name = (d["cls_name"] or "").lower()
        is_gf = class_is_goldfish(cls_name, keywords)

        # If the model is single-class 'fish' and orange gate is enabled, mark orange fish as goldfish
        if use_orange_gate and not is_gf and cls_name == "fish":
            x1, y1, x2, y2 = map(int, d["xyxy"])
            crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
            if orange_ratio(crop) > 0.12:
                is_gf = True

        d["is_goldfish"] = is_gf
        if ignore_other_classes and not is_gf:
            continue
        filtered.append(d)

    # Update TrackStore using (possibly pseudo) track IDs
    store.update_from_dets(filtered, now_t)

    # Zones
    top_h = int(h * (top_zone_pct / 100.0))
    bot_h = int(h * (bottom_zone_pct / 100.0))

    overlay = frame.copy()
    # Draw zones
    cv2.rectangle(overlay, (0, 0), (w, top_h), (255, 0, 0), 2)
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (0, 0, 255), 2)

    candidates = []
    goldfish_count = 0

    # Draw detections + compute decisions per track
    for d in filtered:
        tid = d["tid"]
        if tid is None or tid not in store.tracks:
            continue
        tr = store.tracks[tid]
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cx, cy = d["center"]
        avg_speed = compute_avg_speed(tr, now_t, window_sec)
        at_top = cy <= top_h
        at_bottom = cy >= (h - bot_h)
        likely = is_likely_dead(avg_speed, at_top, at_bottom, speed_thresh)

        cls_name = d["cls_name"]
        if d.get("is_goldfish", False):
            goldfish_count += 1

        color = (0, 255, 0)
        label = f"id={tid} {cls_name} v={avg_speed:.0f}px/s"
        if likely and d.get("is_goldfish", False):
            color = (0, 0, 255)
            label += "  ⚠ likely dead"
            if (now_t - tr.last_alert_t) > 10.0:
                tr.last_alert_t = now_t
                crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(alerts_dir, f"alert_{ts}_id{tid}.jpg")
                cv2.imwrite(out_path, crop)
                candidates.append(out_path)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        # Trajectory (last 20 points)
        pts = list(tr.history)[-20:]
        for i in range(1, len(pts)):
            px1, py1 = int(pts[i - 1][1]), int(pts[i - 1][2])
            px2, py2 = int(pts[i][1]), int(pts[i][2])
            cv2.line(overlay, (px1, py1), (px2, py2), (200, 200, 0), 1)

    preview.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    if candidates:
        with alerts_panel:
            st.subheader("Recent Alert Snapshots")
            cols = st.columns(min(4, len(candidates)))
            for i, p in enumerate(candidates):
                with cols[i % len(cols)]:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)

    with metrics_ph:
        st.markdown(
            f"**Active tracks:** {len(store.tracks)} | "
            f"**Detections (filtered):** {len(filtered)} | "
            f"**Goldfish (tracked):** {goldfish_count} | "
            f"**Top zone:** {top_zone_pct}% | **Bottom zone:** {bottom_zone_pct}%"
        )

    time.sleep(0.01)

st.success("Stopped. Adjust parameters and hit Start again.")
try:
    cap.release()
except Exception:
    pass
