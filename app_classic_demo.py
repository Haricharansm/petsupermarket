# app_classic.py
# Classic "Dead Goldfish Detector" — color (HSV) + motion + simple tracker
# - Upload MP4 is the default source
# - Real-time insights + per-video Tank Session summary persisted as JSON
# - JSON helper handles numpy/deques/Path/etc. to avoid serialization errors

import os
import cv2
import time
import json
import dataclasses
import numpy as np
import streamlit as st
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# =========================
# JSON-safe persistence
# =========================

def _make_jsonable(obj):
    """Recursively convert obj into something json.dump can handle."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pathlib, datetime
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()

    # containers
    if isinstance(obj, (list, tuple, set, deque)):
        return [_make_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _make_jsonable(v) for k, v in obj.items()}

    # dataclasses
    if dataclasses.is_dataclass(obj):
        return _make_jsonable(dataclasses.asdict(obj))

    # fallback
    return str(obj)


def save_session_json(session_obj, out_path="sessions/last_session.json"):
    """Persist a tank/video summary as JSON with bullet-proof conversion."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dumpable = _make_jsonable(session_obj)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dumpable, f, indent=2, ensure_ascii=False)


# =========================
# Tracking & session models
# =========================

@dataclasses.dataclass
class Track:
    tid: int
    history: deque  # of (t, cx, cy)
    last_bbox: Tuple[int, int, int, int]
    last_alert_t: float = 0.0
    alive: bool = True

    def add(self, t: float, cx: float, cy: float, bbox: Tuple[int, int, int, int]):
        self.history.append((t, float(cx), float(cy)))
        self.last_bbox = bbox


@dataclasses.dataclass
class TankSession:
    video_name: str
    started_at: str = dataclasses.field(default_factory=lambda: datetime.utcnow().isoformat())
    total_frames: int = 0
    total_flags: int = 0
    avg_window_s: int = 12
    speed_thresh_px_s: int = 25

    tracked_now: int = 0
    likely_dead_now: int = 0

    snapshots: List[str] = dataclasses.field(default_factory=list)  # saved image paths
    event_log: List[str] = dataclasses.field(default_factory=list)  # strings

    def log(self, msg: str):
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        line = f"• {ts}: {msg}"
        self.event_log.append(line)
        # keep event log size reasonable
        if len(self.event_log) > 400:
            self.event_log = self.event_log[-400:]

    def add_snapshot(self, path: str):
        self.snapshots.append(path)
        # keep only most recent 24 to bound memory/disk usage
        if len(self.snapshots) > 24:
            self.snapshots = self.snapshots[-24:]

    def to_summary(self) -> dict:
        return {
            "video_name": self.video_name,
            "started_at": self.started_at,
            "ended_at": datetime.utcnow().isoformat(),
            "total_frames": int(self.total_frames),
            "total_flags": int(self.total_flags),
            "tracked_now": int(self.tracked_now),
            "likely_dead_now": int(self.likely_dead_now),
            "avg_window_s": int(self.avg_window_s),
            "speed_threshold_px_s": int(self.speed_thresh_px_s),
            "snapshots": list(self.snapshots),
            "event_log": list(self.event_log),
        }


# =========================
# Classic color + motion
# =========================

def hsv_mask(frame_bgr: np.ndarray,
             h_low: int, h_high: int,
             s_low: int, s_high: int,
             v_low: int, v_high: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def detect_blobs(mask: np.ndarray,
                 min_area: int = 60) -> List[Tuple[int, int, int, int]]:
    """Return list of bounding boxes (x1,y1,x2,y2)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes


def box_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pairwise_assign(dets: List[Tuple[int, int, int, int]],
                    tracks: Dict[int, Track],
                    max_dist: float = 60.0) -> Tuple[List[Tuple[int, Tuple]], List[Tuple]]:
    """
    Nearest-neighbor association: returns (matches, new_dets)
    matches: list of (tid, det_bbox)
    new_dets: det_bbox not matched
    """
    unmatched = list(dets)
    matches = []
    if not tracks or not dets:
        return matches, unmatched

    # for each detection, find closest track
    for det in dets:
        cx, cy = box_center(det)
        best_tid = None
        best_d = 1e9
        for tid, tr in tracks.items():
            if not tr.alive or len(tr.history) == 0:
                continue
            _, px, py = tr.history[-1]
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best_d = d
                best_tid = tid
        if best_tid is not None and best_d <= max_dist:
            matches.append((best_tid, det))
            if det in unmatched:
                unmatched.remove(det)

    return matches, unmatched


def avg_speed_px_s(track: Track, window_sec: float) -> float:
    """Average speed (px/s) over the last window_sec of this track history."""
    if len(track.history) < 2:
        return 0.0
    t_end = track.history[-1][0]
    # gather points within window
    pts = [p for p in track.history if p[0] >= (t_end - window_sec)]
    if len(pts) < 2:
        # if short history, approximate using all
        pts = list(track.history)[-2:]
    dist = 0.0
    for i in range(1, len(pts)):
        _, x1, y1 = pts[i - 1]
        _, x2, y2 = pts[i]
        dist += np.hypot(x2 - x1, y2 - y1)
    dt = max(1e-6, pts[-1][0] - pts[0][0])
    return float(dist / dt)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Dead Goldfish Monitor (Classic)", layout="wide")

# Header (clean, product-style)
st.title("🐟 Dead Goldfish Monitor — Classic")
with st.expander("How it works", expanded=True):
    st.markdown(
        """
- Detects orange fish with a tunable **HSV color gate** and tracks motion by ID.
- Uses **average speed** over a sliding window and **top/bottom dwell** zones to flag likely fatalities.
- Maintains a **Tank Session** summary for the full video; saves snapshots and JSON to disk.
        """
    )

# Sidebar — video source first (Upload MP4 default)
with st.sidebar:
    st.header("Video Source")
    source_mode = st.radio("Choose source", ["Upload MP4", "Webcam (0)", "RTSP URL"], index=0)
    rtsp_url = st.text_input("RTSP URL", value="", help="rtsp://user:pass@ip/stream")
    uploaded_file = st.file_uploader("Upload a short video",
                                     type=["mp4", "m4v", "mov", "avi", "mpg", "mpeg4"])

    st.header("Color Gate (HSV)")
    # Sensible defaults for orange
    h_low = st.slider("H low", 0, 179, 5)
    h_high = st.slider("H high", 0, 179, 30)
    s_low = st.slider("S low", 0, 255, 120)
    s_high = st.slider("S high", 0, 255, 255)
    v_low = st.slider("V low", 0, 255, 120)
    v_high = st.slider("V high", 0, 255, 255)

    st.header("Decision Rules")
    window_sec = st.slider("Stillness window (seconds)", 3, 60, 12)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 200, 25)
    top_zone_pct = st.slider("Top zone height (%)", 2, 40, 12)
    bottom_zone_pct = st.slider("Bottom zone height (%)", 2, 40, 12)

    st.header("Run")
    start = st.button("▶︎ Process")
    stop = st.button("⏹ Stop")
    if "running" not in st.session_state:
        st.session_state.running = False
    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

# Prepare file capture
def open_capture():
    if source_mode == "Upload MP4":
        if uploaded_file is None:
            return None
        tmp_path = Path("tmp_upload.mp4")
        with tmp_path.open("wb") as f:
            f.write(uploaded_file.read())
        return cv2.VideoCapture(str(tmp_path)), uploaded_file.name
    elif source_mode == "Webcam (0)":
        return cv2.VideoCapture(0), "webcam_0"
    else:
        return cv2.VideoCapture(rtsp_url), "rtsp_stream"

cap, video_name = open_capture() if st.session_state.running else (None, "")

alerts_dir = Path("alerts")
alerts_dir.mkdir(exist_ok=True)

frame_info = st.empty()
canvas = st.empty()

# Insights area
st.subheader("Insights (live)")
col1, col2, col3, col4 = st.columns(4)
snapshots_ph = st.container()
log_ph = st.container()

# Session + tracker state
if "tracks" not in st.session_state:
    st.session_state.tracks = {}
if "next_tid" not in st.session_state:
    st.session_state.next_tid = 1

def new_tid() -> int:
    t = st.session_state.next_tid
    st.session_state.next_tid += 1
    return t

# Processing loop
session = None
last_t = time.time()
fps_smooth = 0.0

if not st.session_state.running:
    st.info("Select a source and click **Process**.")
elif cap is None or not cap.isOpened():
    st.warning("Provide a valid video source (upload a video, webcam 0, or RTSP).")
    st.session_state.running = False
else:
    session = TankSession(video_name=video_name, avg_window_s=window_sec, speed_thresh_px_s=speed_thresh)
    st.session_state.tracks = {}
    st.session_state.next_tid = 1

    while st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            st.info("End of stream or cannot read more frames.")
            break

        now_t = time.time()
        dt = now_t - last_t
        last_t = now_t
        fps = 1.0 / dt if dt > 1e-3 else 0.0
        fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)

        h, w = frame.shape[:2]
        session.total_frames += 1

        # Mask + detect
        mask = hsv_mask(frame, h_low, h_high, s_low, s_high, v_low, v_high)
        boxes = detect_blobs(mask, min_area=60)

        # Associate to tracks
        matches, new_dets = pairwise_assign(boxes, st.session_state.tracks, max_dist=60)

        # update matched
        for tid, det in matches:
            cx, cy = box_center(det)
            st.session_state.tracks[tid].add(now_t, cx, cy, det)

        # birth new tracks
        for det in new_dets:
            cx, cy = box_center(det)
            tid = new_tid()
            st.session_state.tracks[tid] = Track(
                tid=tid,
                history=deque(maxlen=6000),
                last_bbox=det,
                last_alert_t=0.0,
                alive=True,
            )
            st.session_state.tracks[tid].add(now_t, cx, cy, det)

        # retire stale tracks (not updated for N seconds)
        for tid, tr in list(st.session_state.tracks.items()):
            if len(tr.history) == 0:
                continue
            # if last seen more than 6s ago, retire
            if (now_t - tr.history[-1][0]) > 6.0:
                tr.alive = False

        # Zones & decisions
        top_h = int(h * (top_zone_pct / 100.0))
        bot_h = int(h * (bottom_zone_pct / 100.0))
        overlay = frame.copy()

        # Draw zones
        cv2.rectangle(overlay, (0, 0), (w, top_h), (255, 0, 0), 2)
        cv2.rectangle(overlay, (0, h - bot_h), (w, h), (0, 0, 255), 2)

        likely_dead_now = 0
        tracked_now = 0

        for tid, tr in st.session_state.tracks.items():
            if not tr.alive or len(tr.history) == 0:
                continue
            tracked_now += 1
            x1, y1, x2, y2 = tr.last_bbox
            cx, cy = box_center(tr.last_bbox)
            v = avg_speed_px_s(tr, window_sec)

            at_top = cy <= top_h
            at_bottom = cy >= (h - bot_h)
            likely = (v <= speed_thresh) and (at_top or at_bottom)
            if likely:
                likely_dead_now += 1

            color = (0, 255, 0)
            label = f"id={tid} v={int(v)}px/s"
            if likely:
                color = (0, 0, 255)
                label += " ⚠"

                # throttle snapshot frequency per track
                if (now_t - tr.last_alert_t) > 8.0:
                    tr.last_alert_t = now_t
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
                    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                    out_path = alerts_dir / f"alert_{ts}_id{tid}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    session.add_snapshot(str(out_path))
                    session.total_flags += 1
                    session.log(f"Track {tid} flagged (v≈{int(v)}px/s).")

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # draw brief trail (last ~20 points)
            pts = list(tr.history)[-20:]
            for i in range(1, len(pts)):
                _, px1, py1 = pts[i - 1]
                _, px2, py2 = pts[i]
                cv2.line(overlay, (int(px1), int(py1)), (int(px2), int(py2)), (200, 200, 0), 1)

        session.tracked_now = tracked_now
        session.likely_dead_now = likely_dead_now

        frame_info.write(f"Frame: {w}×{h}  |  FPS: {fps_smooth:.1f}  |  Tracks: {tracked_now}")
        canvas.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Insights UI (live)
        summary_line = (
            f"{likely_dead_now} fish look likely dead or distressed "
            f"(low motion in {likely_dead_now} near the top/bottom)."
            if likely_dead_now > 0
            else "No fish currently flagged for low motion in top/bottom zones."
        )
        st.markdown(summary_line)
        col1.metric("Tracked fish (now)", tracked_now)
        col2.metric("Likely dead (now)", likely_dead_now)
        col3.metric("Avg window (s)", window_sec)
        col4.metric("Speed threshold", f"{speed_thresh} px/s")

        # Snapshots
        with snapshots_ph:
            if session.snapshots:
                st.caption("Recent alert snapshots")
                cols = st.columns(3)
                for i, p in enumerate(session.snapshots[-3:][::-1]):
                    with cols[i % 3]:
                        st.image(p, use_column_width=True)

        # Event log
        with log_ph:
            if session.event_log:
                st.caption("Event log")
                for line in session.event_log[-20:][::-1]:
                    st.write(line)

        # small wait to keep UI fluid
        time.sleep(0.01)

    # after loop
    try:
        cap.release()
    except Exception:
        pass

    if session:
        # persist a compact per-video summary
        save_session_json(session.to_summary(), f"sessions/{Path(session.video_name).stem}_summary.json")
        st.success("Done. Open the **alerts/** folder for snapshots and **sessions/** for JSON summaries.")
