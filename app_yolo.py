# app_yolo.py — Streamlit app using Ultralytics YOLO + ByteTrack
# - Detects fish (expects a model trained with a "goldfish" class)
# - Tracks with ByteTrack (persist=True)
# - Applies stillness + top/bottom dwell logic per tracked goldfish
# Run:
#   pip install -r requirements.txt
#   streamlit run app_yolo.py
import os
import time
from typing import List, Dict, Tuple

import cv2
import numpy as np
import streamlit as st

from yolo_infer import YOLOTracker
from track_store import TrackStore, Track
from decision import compute_avg_speed, is_likely_dead

st.set_page_config(page_title="Dead Goldfish Detector (YOLO)", layout="wide")
st.title("🐟 Dead Goldfish Detector — YOLO + ByteTrack")

st.markdown(
    """
This scaffold uses **Ultralytics YOLO** (detection) + **ByteTrack** (tracking) and then applies a **simple rule** to flag
**likely dead** goldfish:
- Nearly **motionless** for *N* seconds, **and**
- Dwelling at the **top** (floating) or **bottom** (sunk) zone of the tank.

👉 Train your model with a `goldfish` class (and optionally `other_fish`, `debris`) and point the app to `best.pt`.
"""
)

# ---------------------
# Sidebar controls
# ---------------------
with st.sidebar:
    st.header("Model")
    model_path = st.text_input("YOLO model path (.pt)", value="best.pt", help="Path to trained weights (e.g., runs/detect/train/weights/best.pt)")
    conf = st.slider("Confidence threshold", 0.05, 0.8, 0.25, 0.01)
    iou = st.slider("NMS IoU", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.selectbox("Image size", [480, 512, 640, 960, 1280], index=2)
    tracker_alg = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)
    device = st.selectbox("Device", ["auto", "cpu"], index=0, help="Use 'auto' for GPU if available, else CPU")

    st.header("Classes")
    st.write("Which detected classes should be treated as **goldfish** for dead/alive logic?")
    goldfish_class_keywords = st.text_input("Goldfish class keywords (comma-separated)", value="goldfish")
    ignore_other_classes = st.checkbox("Ignore non-goldfish classes", value=True)

    st.header("Video Source")
    source_mode = st.radio("Choose source", ["Webcam (0)", "RTSP URL", "Upload MP4"], index=0)
    rtsp_url = st.text_input("RTSP URL", value="", help="e.g., rtsp://user:pass@ip/stream")
    uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi"])

    st.header("Decision Rules")
    window_sec = st.slider("Stillness window (seconds)", 5, 180, 30)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 300, 25)
    top_zone_pct = st.slider("Top zone height (%)", 2, 40, 12)
    bottom_zone_pct = st.slider("Bottom zone height (%)", 2, 40, 12)

    st.header("Run")
    start = st.button("▶️ Start")
    stop = st.button("⏹ Stop")
    if "running" not in st.session_state:
        st.session_state.running = False
    if start: st.session_state.running = True
    if stop: st.session_state.running = False

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
        tmp_path = "tmp_upload.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp_path)
    if not cap or not cap.isOpened():
        return None
    return cap

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def class_is_goldfish(name: str, keywords: List[str]) -> bool:
    n = (name or "").lower()
    return any(kw.strip().lower() in n for kw in keywords if kw.strip())

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

# Load YOLO
try:
    yolo = YOLOTracker(model_path=model_path, conf=conf, iou=iou, imgsz=imgsz, tracker=tracker_alg, device=None if device=="auto" else device)
except Exception as e:
    st.error(f"Failed to load YOLO model from '{model_path}': {e}")
    st.stop()

store = TrackStore(maxlen=6000, stale_sec=12)

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
    # Optionally filter to only "goldfish" classes for logic
    filtered = []
    for d in dets:
        is_gf = class_is_goldfish(d["cls_name"], keywords)
        if ignore_other_classes and not is_gf:
            continue
        d["is_goldfish"] = is_gf
        filtered.append(d)

    # Update TrackStore using YOLO track IDs
    store.update_from_dets(filtered, now_t)

    # Zones
    top_h = int(h * (top_zone_pct/100.0))
    bot_h = int(h * (bottom_zone_pct/100.0))

    overlay = frame.copy()
    # Draw zones
    cv2.rectangle(overlay, (0,0), (w, top_h), (255, 0, 0), 2)
    cv2.rectangle(overlay, (0,h-bot_h), (w, h), (0, 0, 255), 2)

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

        color = (0,255,0)
        label = f"id={tid} {cls_name} v={avg_speed:.0f}px/s"
        if likely and d.get("is_goldfish", False):
            color = (0,0,255)
            label += "  ⚠ likely dead"
            if (now_t - tr.last_alert_t) > 10.0:
                tr.last_alert_t = now_t
                crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(alerts_dir, f"alert_{ts}_id{tid}.jpg")
                cv2.imwrite(out_path, crop)
                candidates.append(out_path)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Trajectory (last 20 points)
        pts = list(tr.history)[-20:]
        for i in range(1, len(pts)):
            px1,py1 = int(pts[i-1][1]), int(pts[i-1][2])
            px2,py2 = int(pts[i][1]), int(pts[i][2])
            cv2.line(overlay, (px1,py1), (px2,py2), (200,200,0), 1)

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
