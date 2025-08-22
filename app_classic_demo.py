# app_classic.py
# Goldfish tank health monitor (classic color+motion)
# - HSV color mask to isolate orange fish
# - Lightweight ID tracker with sliding-window speed
# - Flags likely fatalities (low motion + top/bottom dwell)
# - Maintains persistent "Tank Session" across the whole video
# - Exports JSON/CSV + narrative summary

import os
import cv2
import time
import json
import math
import io
import csv
import platform
import numpy as np
from collections import deque
import streamlit as st

# -----------------------
# Page & styling
# -----------------------
st.set_page_config(page_title="Goldfish Tank Health Monitor", layout="wide")
st.markdown(
    """
    <style>
      .metrics {display:flex; gap:2rem; margin:.25rem 0 .75rem 0}
      .metric {font-size:34px; font-weight:700; line-height:1.1}
      .metric-label {font-size:13px; color:#6b7280}
      .capsule {background:#F8FAFC;border:1px solid #E5E7EB;border-radius:12px;padding:14px 16px}
      .pill {background:#EFF6FF;border:1px solid #DBEAFE;border-radius:999px;padding:2px 10px; font-size:12px}
      .event {font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono";}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Safe OpenCV import diag
# -----------------------
try:
    _ = cv2.UMat  # touch cv2 to ensure loaded
except Exception as e:
    st.error("OpenCV failed to import. Details below.")
    st.code(
        f"Python={platform.python_version()}\n"
        f"Platform={platform.platform()}\n"
        f"Error: {repr(e)}"
    )
    st.stop()


# -----------------------
# Utility
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_video_key(mode: str, file, rtsp: str) -> str:
    if mode == "Upload MP4" and file is not None:
        return f"upload:{file.name}:{file.size}"
    if mode == "RTSP URL" and rtsp:
        return f"rtsp:{rtsp}"
    return f"webcam:0"


def _sessions_dir():
    d = "sessions"
    ensure_dir(d)
    return d


def save_session_json(tank):
    path = os.path.join(_sessions_dir(), f"{tank['key']}.json")
    dumpable = {
        **tank,
        "unique_tracks": list(tank["unique_tracks"]),
        "ever_flagged": list(tank["ever_flagged"]),
        "top_flagged": list(tank["top_flagged"]),
        "bottom_flagged": list(tank["bottom_flagged"]),
        "events": list(tank["events"]),
    }
    with open(path, "w") as f:
        json.dump(dumpable, f)


def load_session_json(key):
    path = os.path.join(_sessions_dir(), f"{key}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    data["unique_tracks"] = set(data.get("unique_tracks", []))
    data["ever_flagged"] = set(data.get("ever_flagged", []))
    data["top_flagged"] = set(data.get("top_flagged", []))
    data["bottom_flagged"] = set(data.get("bottom_flagged", []))
    data["events"] = deque(data.get("events", []), maxlen=50)
    return data


def init_tank_session(key: str) -> dict:
    return {
        "key": key,
        "started_at": time.time(),
        "frames": 0,
        "unique_tracks": set(),
        "ever_flagged": set(),
        "top_flagged": set(),
        "bottom_flagged": set(),
        "snapshots": deque(maxlen=12),  # paths
        "events": deque(maxlen=50),     # text
    }


def open_capture(mode: str, rtsp: str, uploaded):
    if mode == "Upload MP4":
        if uploaded is None:
            return None
        tmp_path = "tmp_upload.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        return cv2.VideoCapture(tmp_path)
    if mode == "RTSP URL":
        return cv2.VideoCapture(rtsp)
    # Webcam 0
    return cv2.VideoCapture(0)


# -----------------------
# Basic tracker (nearest-neighbor)
# -----------------------
class TrackState:
    __slots__ = ("id", "last", "t_last", "history", "box", "stale")

    def __init__(self, tid, cx, cy, t, box):
        self.id = tid
        self.last = (cx, cy)
        self.t_last = t
        self.history = deque([(t, cx, cy)], maxlen=1800)  # ~minutes
        self.box = box
        self.stale = 0


class SimpleTracker:
    def __init__(self, max_dist=60, max_stale=20):
        self.next_id = 1
        self.tracks = {}
        self.max_dist = max_dist
        self.max_stale = max_stale

    def update(self, detections, t):
        """
        detections: list of (cx, cy, (x1,y1,x2,y2))
        returns list of dicts per detection:
          {tid, cx, cy, box}
        """
        assigned = set()
        out = []

        # Try to match detections to existing tracks
        for tid, st in self.tracks.items():
            st.stale += 1

        for i, (cx, cy, box) in enumerate(detections):
            best_tid, best_d = None, 1e9
            for tid, st in self.tracks.items():
                d = math.hypot(cx - st.last[0], cy - st.last[1])
                if d < best_d and d <= self.max_dist:
                    best_d, best_tid = d, tid
            if best_tid is None:
                # new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackState(tid, cx, cy, t, box)
                assigned.add(tid)
            else:
                tid = best_tid
                assigned.add(tid)
                st = self.tracks[tid]
                st.last = (cx, cy)
                st.t_last = t
                st.history.append((t, cx, cy))
                st.box = box
                st.stale = 0

            out.append({"tid": tid, "cx": cx, "cy": cy, "box": box})

        # Cull stale
        to_del = [tid for tid, st in self.tracks.items() if st.stale > self.max_stale]
        for tid in to_del:
            del self.tracks[tid]

        return out

    def avg_speed(self, tid, window_sec=12):
        st = self.tracks.get(tid)
        if not st or len(st.history) < 2:
            return 0.0
        hist = list(st.history)
        # keep only last N seconds
        t_now = hist[-1][0]
        pts = [(t, x, y) for (t, x, y) in hist if t_now - t <= window_sec]
        if len(pts) < 2:
            return 0.0
        dist = 0.0
        dt = pts[-1][0] - pts[0][0]
        for i in range(1, len(pts)):
            dist += math.hypot(pts[i][1] - pts[i-1][1], pts[i][2] - pts[i-1][2])
        return (dist / dt) if dt > 0 else 0.0


# -----------------------
# Image ops
# -----------------------
def mask_orange(bgr, h_low, s_low, show_mask=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # One broad band for orange/yellow. Tune as needed.
    lower = np.array([h_low, s_low, 80], dtype=np.uint8)
    upper = np.array([30 + h_low, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    # Morphology for smoother blobs
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    if show_mask:
        return m, hsv
    return m, None


def find_centroids(mask, min_area=120):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2.0
        cy = y + h / 2.0
        out.append((cx, cy, (x, y, x + w, y + h)))
    return out


def summarize_live(tracked_now, flagged_now, at_top, at_bottom):
    if flagged_now > 0:
        zone = []
        if at_top > 0:
            zone.append(f"{at_top} near the surface")
        if at_bottom > 0:
            zone.append(f"{at_bottom} near the bottom")
        zone_txt = ", ".join(zone)
        return f"{flagged_now} fish look likely dead or distressed ({zone_txt})."
    if tracked_now > 0:
        return f"All {tracked_now} fish look alive and active right now."
    return "Fish detected but building motion history…"


# -----------------------
# Sidebar (Controls)
# -----------------------
with st.sidebar:
    st.header("Video Source")
    source_mode = st.radio("Choose source", ["Upload MP4", "Webcam (0)", "RTSP URL"], index=0)
    rtsp_url = st.text_input("RTSP URL", value="") if source_mode == "RTSP URL" else ""
    uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "m4v", "mpg", "mpeg4"])

    st.header("Color Gate (HSV)")
    h_low = st.slider("H low", 0, 179, 5)
    s_low = st.slider("S low", 0, 255, 120)

    st.header("Decision Rules")
    window_sec = st.slider("Stillness window (seconds)", 3, 60, 12)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 200, 25)
    top_zone_pct = st.slider("Top zone height (%)", 2, 40, 12)
    bottom_zone_pct = st.slider("Bottom zone height (%)", 2, 40, 12)

    st.header("Run")
    start = st.button("▶ Process", use_container_width=True)
    stop = st.button("■ Stop", use_container_width=True)
    if "running" not in st.session_state:
        st.session_state.running = False
    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False

st.title("Goldfish Tank Health Monitor — Classic")

st.markdown(
    """
    <div class="capsule">
      <ul>
        <li>Detects orange fish (tunable HSV gate) and tracks movement</li>
        <li>Per-fish <b>speed</b> and <b>dwell-zone</b> logic to flag likely fatalities</li>
        <li>Maintains a <b>Tank Session</b> across the entire video with an overall verdict</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

frame_meta = st.empty()
preview = st.empty()

# -----------------------
# Initialize / resume Tank Session
# -----------------------
vid_key = compute_video_key(source_mode, uploaded_file, rtsp_url)
if "tank" not in st.session_state or st.session_state["tank"]["key"] != vid_key:
    loaded = load_session_json(vid_key)
    st.session_state["tank"] = loaded if loaded else init_tank_session(vid_key)
tank = st.session_state["tank"]

alerts_dir = "alerts"
ensure_dir(alerts_dir)

# -----------------------
# Capture
# -----------------------
cap = open_capture(source_mode, rtsp_url, uploaded_file)
if not cap or not cap.isOpened():
    st.info("Provide a valid video source (upload a short MP4, use webcam 0, or provide an RTSP URL).")
    st.stop()

# -----------------------
# Process loop
# -----------------------
tracker = SimpleTracker(max_dist=70, max_stale=15)
last_t = time.time()
fps_smooth = 0.0

show_mask_opt = st.checkbox("Show mask")

insights_anchor = st.empty()
insight_gallery_ph = st.container()
event_log_ph = st.container()

while st.session_state.running:
    ok, frame = cap.read()
    if not ok:
        st.warning("End of stream or cannot read more frames.")
        break

    now_t = time.time()
    dt = max(1e-3, now_t - last_t)
    last_t = now_t
    fps = 1.0 / dt
    fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)

    h, w = frame.shape[:2]

    # Zones
    top_h = int(h * (top_zone_pct / 100.0))
    bot_h = int(h * (bottom_zone_pct / 100.0))

    # Color mask -> centroids
    mask, _ = mask_orange(frame, h_low, s_low, show_mask=show_mask_opt)
    dets = find_centroids(mask)

    # Update tracker
    det_list = [(cx, cy, box) for (cx, cy, box) in dets]
    assigned = tracker.update(det_list, now_t)

    # Draw base
    overlay = frame.copy()
    # Zone lines
    cv2.rectangle(overlay, (0, 0), (w - 1, top_h), (255, 0, 0), 2)
    cv2.rectangle(overlay, (0, h - bot_h), (w - 1, h - 1), (0, 0, 255), 2)

    tracked_now = len(assigned)
    flagged_now = 0
    flagged_top = 0
    flagged_bottom = 0

    # Manage per-track logic + paint
    for d in assigned:
        tid, cx, cy, (x1, y1, x2, y2) = d["tid"], d["cx"], d["cy"], d["box"]
        tank["unique_tracks"].add(tid)

        v = tracker.avg_speed(tid, window_sec)
        at_top = cy <= top_h
        at_bottom = cy >= (h - bot_h)
        likely = v < speed_thresh and (at_top or at_bottom)

        color = (0, 255, 0)
        label = f"id={tid} v={int(v)}px/s"
        if likely:
            color = (0, 0, 255)
            label += "  ⚠"
            flagged_now += 1
            if at_top:
                flagged_top += 1
            if at_bottom:
                flagged_bottom += 1

            # alert snapshots (rate-limited by frames)
            if tank["frames"] % 7 == 0:
                crop = overlay[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
                ts = time.strftime("%Y%m%d-%H%M%S")
                pth = os.path.join(alerts_dir, f"alert_{ts}_id{tid}.jpg")
                try:
                    cv2.imwrite(pth, crop)
                    tank["snapshots"].appendleft(pth)
                except Exception:
                    pass

            # session counters/events
            tank["ever_flagged"].add(tid)
            if at_top:
                tank["top_flagged"].add(tid)
            if at_bottom:
                tank["bottom_flagged"].add(tid)
            if tank["frames"] % 15 == 0:
                tank["events"].appendleft(f"{time.strftime('%Y%m%d-%H%M%S')}: Track {tid} flagged (v={int(v)}px/s).")

        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(overlay, label, (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # breadcrumbs
        stt = tracker.tracks.get(tid)
        if stt:
            pts = list(stt.history)[-20:]
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][1]), int(pts[i-1][2]))
                p2 = (int(pts[i][1]), int(pts[i][2]))
                cv2.line(overlay, p1, p2, (200, 200, 0), 1)

    # Show frame
    frame_meta.write(f"Frame: {w}×{h} | FPS: {fps_smooth:.1f} | Tracks: {len(tracker.tracks)}")
    if show_mask_opt:
        # Compose side-by-side mask+overlay
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([overlay, mask_bgr])
        preview.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    else:
        preview.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # --- Live insights ---
    msg = summarize_live(tracked_now, flagged_now, flagged_top, flagged_bottom)
    with insights_anchor.container():
        st.subheader("Insights (live)")
        st.write(msg)
        cols = st.columns(4)
        cols[0].metric("Tracked fish (now)", tracked_now)
        cols[1].metric("Likely dead (now)", flagged_now)
        cols[2].metric("Avg window (s)", window_sec)
        cols[3].metric("Speed threshold", f"{speed_thresh} px/s")

    # Gallery of recent alert snapshots
    with insight_gallery_ph:
        st.markdown("**Recent alert snapshots**")
        if len(tank["snapshots"]) == 0:
            st.caption("No snapshots yet.")
        else:
            cols = st.columns(3)
            for i, p in enumerate(list(tank["snapshots"])[:9]):
                with cols[i % 3]:
                    st.image(p, use_column_width=True)

    # Event log
    with event_log_ph:
        st.markdown("**Event log**")
        if len(tank["events"]) == 0:
            st.caption("No events yet.")
        else:
            for e in list(tank["events"])[:12]:
                st.markdown(f"<div class='event'>• {e}</div>", unsafe_allow_html=True)

    # Advance session counters and autosave
    tank["frames"] += 1
    if tank["frames"] % 20 == 0:
        save_session_json(tank)

    time.sleep(0.005)

# Done / finalize
try:
    cap.release()
except Exception:
    pass

save_session_json(tank)
st.success("Processing completed. Open the 'alerts' and 'sessions' folders for artifacts.")


# -----------------------
# Tank Session (roll-up) + downloads
# -----------------------
from datetime import datetime

session_time = max(0, int(time.time() - tank["started_at"]))
n_unique = len(tank["unique_tracks"])
n_flagged = len(tank["ever_flagged"])
n_top = len(tank["top_flagged"])
n_bottom = len(tank["bottom_flagged"])

st.subheader("Tank Session (so far)")
cols = st.columns(5)
cols[0].metric("Unique fish (session)", n_unique)
cols[1].metric("Ever flagged", n_flagged)
cols[2].metric("Surface flagged", n_top)
cols[3].metric("Bottom flagged", n_bottom)
cols[4].metric("Session time", f"{session_time}s")

# Narrative
narrative = (
    f"As of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
    f"we’ve observed {n_unique} unique fish over {session_time}s. "
    + (f"{n_flagged} fish were flagged at least once "
       f"({n_top} near the surface, {n_bottom} near the bottom). "
       if n_flagged else "No fatalities are suspected. ")
    + "Live monitoring continues."
)
st.markdown("**Session narrative**")
st.write(narrative)

# Download JSON
report_json = {
    "generated_at": datetime.now().isoformat(),
    "session_seconds": session_time,
    "unique_fish": n_unique,
    "ever_flagged": n_flagged,
    "flagged_surface": n_top,
    "flagged_bottom": n_bottom,
    "events": list(tank["events"]),
    "narrative": narrative,
}
st.download_button(
    "Download session JSON",
    data=json.dumps(report_json, indent=2),
    file_name="tank_session.json",
    mime="application/json",
    use_container_width=True,
)

# Download CSV (events)
csv_buf = io.StringIO()
writer = csv.writer(csv_buf)
writer.writerow(["timestamp_event"])
for e in list(tank["events"]):
    writer.writerow([e])
st.download_button(
    "Download event log CSV",
    data=csv_buf.getvalue().encode(),
    file_name="tank_events.csv",
    mime="text/csv",
    use_container_width=True,
)
