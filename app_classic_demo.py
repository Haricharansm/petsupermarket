# classic_yolo.py
# Streamlit app: Classic (color + motion) "Dead Goldfish Detector" + natural-language insights
# Run:
#   pip install streamlit opencv-python-headless numpy
#   streamlit run classic_yolo.py

import os
import time
from collections import deque
from statistics import median

import cv2
import numpy as np
import streamlit as st


# ============================
# Small, self-contained tracker
# ============================

class Track:
    """Simple centroid track with time-stamped history: (t, cx, cy)."""
    def __init__(self, tid, cx, cy, t_now, maxlen=600):
        self.id = tid
        self.history = deque(maxlen=maxlen)  # (t, cx, cy)
        self.last_t = t_now
        self.add_point(t_now, cx, cy)

    def add_point(self, t, cx, cy):
        self.history.append((t, float(cx), float(cy)))
        self.last_t = t

class Tracker:
    """Nearest-neighbor association on centroids with a gating distance."""
    def __init__(self, max_dist=60.0, stale_sec=2.5, maxlen=600):
        self.max_dist = float(max_dist)
        self.stale_sec = float(stale_sec)
        self.maxlen = int(maxlen)
        self.tracks = {}      # tid -> Track
        self._next_id = 1

    def _distance(self, a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _spawn(self, cx, cy, t_now):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(tid, cx, cy, t_now, self.maxlen)
        return tid

    def _prune_stale(self, t_now):
        stale = [tid for tid, tr in self.tracks.items() if t_now - tr.last_t > self.stale_sec]
        for tid in stale:
            self.tracks.pop(tid, None)

    def update(self, detections, t_now):
        """
        detections: list of dicts with keys: 'cx','cy','bbox'
        Returns the same detections augmented with 'tid'.
        """
        self._prune_stale(t_now)

        # Build candidate pairs (det, track) within gate
        det_indices = list(range(len(detections)))
        track_items = list(self.tracks.items())  # [(tid, tr), ...]
        pairs = []
        for di in det_indices:
            cx, cy = detections[di]['cx'], detections[di]['cy']
            for tid, tr in track_items:
                # use most recent track point as target
                _, tx, ty = tr.history[-1]
                d = self._distance((cx, cy), (tx, ty))
                if d <= self.max_dist:
                    pairs.append((d, di, tid))
        # Greedy assignment by distance
        pairs.sort(key=lambda x: x[0])
        used_det = set()
        used_tid = set()
        for d, di, tid in pairs:
            if di in used_det or tid in used_tid:
                continue
            used_det.add(di)
            used_tid.add(tid)
            # assign
            cx, cy = detections[di]['cx'], detections[di]['cy']
            self.tracks[tid].add_point(t_now, cx, cy)
            detections[di]['tid'] = tid

        # Spawn new tracks for unmatched dets
        for di in det_indices:
            if di not in used_det:
                cx, cy = detections[di]['cx'], detections[di]['cy']
                tid = self._spawn(cx, cy, t_now)
                detections[di]['tid'] = tid

        return detections


# ============================
# Classic color+motion detector
# ============================

def color_mask(frame_bgr, h_low, h_high, s_low, s_high, v_low, v_high):
    """HSV mask for orange-y fish."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def detect_from_mask(mask, min_area=150):
    """Return list of detections [{'cx','cy','bbox'}] from a binary mask."""
    # clean-up
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    # contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        dets.append(dict(cx=cx, cy=cy, bbox=(x,y,x+w,y+h)))
    return dets


# ============================
# Insights (natural language)
# ============================

def avg_speed_px_s(tr: Track, now_t: float, win_sec: float) -> float:
    pts = [p for p in tr.history if now_t - p[0] <= win_sec]
    if len(pts) < 2: return 0.0
    dist = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][1] - pts[i-1][1]
        dy = pts[i][2] - pts[i-1][2]
        dist += float(np.hypot(dx, dy))
    dt = pts[-1][0] - pts[0][0]
    return dist / dt if dt > 1e-6 else 0.0

def status_for_track(avg_v: float, at_top: bool, at_bottom: bool,
                     speed_thresh: float, window_sec: float):
    low_motion = avg_v <= speed_thresh
    if low_motion and (at_top or at_bottom):
        where = "top" if at_top else "bottom"
        return ("likely_dead", f"still ~{int(window_sec)}s near **{where}**")
    if low_motion:
        return ("resting", f"still ~{int(window_sec)}s (mid-water)")
    return ("alive", f"active (v≈{avg_v:.0f}px/s)")

def render_insights(tracker: Tracker, now_t: float, h: int, top_h: int, bot_h: int,
                    window_sec: float, speed_thresh: float) -> str:
    speeds, dead_lines, rest_lines = [], [], []
    for tid, tr in list(tracker.tracks.items()):
        avg_v = avg_speed_px_s(tr, now_t, window_sec)
        speeds.append(avg_v)
        if not tr.history: continue
        _, cx, cy = tr.history[-1]
        at_top = cy <= top_h
        at_bottom = cy >= (h - bot_h)
        status, why = status_for_track(avg_v, at_top, at_bottom, speed_thresh, window_sec)
        if status == "likely_dead":
            dead_lines.append(f"• **Fish #{tid}** → ⚠️ *likely dead*: {why} (v≈{avg_v:.0f}px/s)")
        elif status == "resting":
            rest_lines.append(f"• Fish #{tid} → 💤 *resting*: {why} (v≈{avg_v:.0f}px/s)")

    n_tracks = len(tracker.tracks)
    n_dead, n_rest = len(dead_lines), len(rest_lines)
    med_v = median(speeds) if speeds else 0.0
    max_v = max(speeds) if speeds else 0.0

    if n_dead > 0:
        headline = f"**Now:** {n_tracks} fish | **{n_dead} likely dead**, {n_rest} resting | median v≈{med_v:.0f}px/s, max v≈{max_v:.0f}px/s"
    elif n_rest > 0:
        headline = f"**Now:** {n_tracks} fish | none likely dead, **{n_rest} resting** | median v≈{med_v:.0f}px/s, max v≈{max_v:.0f}px/s"
    else:
        headline = f"**Now:** {n_tracks} fish | ✅ all look active | median v≈{med_v:.0f}px/s, max v≈{max_v:.0f}px/s"

    md = [headline, ""]
    if dead_lines:
        md += ["### ⚠️ Likely dead", *dead_lines, ""]
    if rest_lines:
        md += ["### 💤 Possibly resting (low motion, not top/bottom)", *rest_lines, ""]
    return "\n".join(md)


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="Dead Goldfish Detector — Classic", layout="wide")
st.title("🐟 Dead Goldfish Detector — Classic")

st.markdown("""
This mode uses **color + motion** and a **simple tracker** to flag *likely dead* goldfish:

- Detect **orange** fish via HSV color mask  
- Track centroids across frames  
- If **avg speed** is low for *N* seconds **and** the fish dwells near the **top** or **bottom**, raise an alert  

Use this mode standalone or alongside a trained YOLO model.
""")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Video Source")

    # Default to Upload MP4; remember last choice
    if "source_mode_classic" not in st.session_state:
        st.session_state.source_mode_classic = "Upload MP4"

    source_options = ["Upload MP4", "Webcam (0)"]
    source_mode = st.radio(
        "Choose source",
        source_options,
        index=source_options.index(st.session_state.source_mode_classic),
        key="source_mode_classic",
    )

    uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi"])
    st.caption("Tip: small 10–20s clips run fastest.")

    st.header("Color Gate (HSV)")
    # Defaults target “orange”: roughly 5–25 hue range
    h_low  = st.slider("H low",  0, 179, 5)
    h_high = st.slider("H high", 0, 179, 25)
    s_low  = st.slider("S low",  0, 255, 120)
    s_high = st.slider("S high", 0, 255, 255)
    v_low  = st.slider("V low",  0, 255, 80)
    v_high = st.slider("V high", 0, 255, 255)

    show_mask = st.checkbox("Show mask", value=False)

    st.header("Decision Rules")
    window_sec   = st.slider("Stillness window (seconds)", 5, 120, 30)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 300, 25)
    top_zone_pct = st.slider("Top zone height (%)", 2, 40, 12)
    bot_zone_pct = st.slider("Bottom zone height (%)", 2, 40, 12)

    st.header("Tracker")
    max_match_dist = st.slider("Max match distance (px)", 10, 150, 70)
    min_area = st.slider("Min blob area (px)", 50, 4000, 150)
    stale_sec = st.slider("Track stale timeout (s)", 1, 10, 3)

    st.header("Run")
    start = st.button("▶️ Start")
    stop = st.button("⏹ Stop")
    if "running" not in st.session_state:
        st.session_state.running = False
    if start: st.session_state.running = True
    if stop: st.session_state.running = False

# --- Placeholders ---
frame_info_ph = st.empty()
preview_ph    = st.empty()
insights_ph   = st.container()

def open_capture():
    if source_mode == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    else:
        if uploaded_file is None:
            return None
        tmp_path = "tmp_upload_video.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp_path)
    if not cap or not cap.isOpened():
        return None
    return cap

# Main loop
cap = open_capture()
if not cap:
    st.warning("Provide a valid source (upload a video or use Webcam 0).")
    st.stop()

tracker = Tracker(max_dist=max_match_dist, stale_sec=stale_sec)
fps_smooth = 0.0
last_t = time.time()

while st.session_state.running:
    ok, frame = cap.read()
    if not ok:
        st.info("End of stream or cannot read more frames.")
        break

    t_now = time.time()
    dt = t_now - last_t
    last_t = t_now
    fps = 1.0 / dt if dt > 1e-3 else 0.0
    fps_smooth = 0.85 * fps_smooth + 0.15 * fps if fps_smooth > 0 else fps

    h, w = frame.shape[:2]
    top_h = int(h * (top_zone_pct/100.0))
    bot_h = int(h * (bot_zone_pct/100.0))

    # Mask & detections
    mask = color_mask(frame, h_low, h_high, s_low, s_high, v_low, v_high)
    dets = detect_from_mask(mask, min_area=min_area)

    # Track
    dets = tracker.update(dets, t_now)

    # Draw overlay
    overlay = frame.copy()
    # zones
    cv2.rectangle(overlay, (0,0), (w, top_h), (255, 0, 0), 2)
    cv2.rectangle(overlay, (0,h-bot_h), (w, h), (0, 0, 255), 2)

    # visualize dets & tracks
    for d in dets:
        x1,y1,x2,y2 = d['bbox']
        cx, cy = int(d['cx']), int(d['cy'])
        tid = d['tid']
        tr = tracker.tracks.get(tid)
        # speed over window for live label
        avg_v = avg_speed_px_s(tr, t_now, window_sec) if tr else 0.0
        at_top = cy <= top_h
        at_bottom = cy >= (h - bot_h)
        status, _ = status_for_track(avg_v, at_top, at_bottom, speed_thresh, window_sec)

        color = (0,255,0)  # alive
        if status == "resting":      color = (0,200,255)
        elif status == "likely_dead": color = (0,0,255)

        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        label = f"#{tid} v={avg_v:.0f}px/s"
        cv2.putText(overlay, label, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # short trajectory
        if tr and len(tr.history) > 1:
            pts = list(tr.history)[-20:]
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][1]), int(pts[i-1][2]))
                p2 = (int(pts[i][1]),   int(pts[i][2]))
                cv2.line(overlay, p1, p2, (200,200,0), 1)

        cv2.circle(overlay, (cx,cy), 2, (0,255,255), -1)

    # Show frame or mask
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else overlay
    frame_info_ph.write(f"Frame: {w}×{h} | FPS: {fps_smooth:.1f} | Tracks: {len(tracker.tracks)}")
    preview_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Insights
    insights_md = render_insights(tracker, t_now, h, top_h, bot_h, window_sec, speed_thresh)
    with insights_ph:
        st.subheader("Insights")
        st.markdown(insights_md)

    # keep UI responsive
    time.sleep(0.005)

# cleanup
try:
    cap.release()
except Exception:
    pass

st.success("Stopped. Adjust parameters and hit Start again.")
