# app_classic.py
# Goldfish Health Monitor — Classic (color + motion + lightweight tracking)
# Run: streamlit run app_classic.py

import os, time, math, cv2, numpy as np
from collections import deque
from dataclasses import dataclass, field
import streamlit as st

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# =========================
# UI —— Elegant landing
# =========================
st.set_page_config(
    page_title="Goldfish Health Monitor",
    page_icon="🐟",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .card {
        background: var(--background-color);
        border: 1px solid rgba(0,0,0,.06);
        border-radius: 14px; padding: 18px 20px;
      }
      .pill {
        display:inline-block; padding:6px 10px; border-radius:100px;
        border:1px solid rgba(0,0,0,.07); margin-right:8px; font-size:0.9rem;
        background: rgba(0,0,0,.02);
      }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([0.75, 0.25])
with left:
    st.markdown(
        """
        # 🐟 Goldfish Health Monitor — Classic
        Real-time monitoring of aquarium fish using **color**, **motion**, and a lightweight tracker.
        """.strip()
    )
    st.markdown(
        """
        <div class="card">
          <ul style="margin:0 0 0 1.1rem; line-height:1.8;">
            <li>Detects orange fish (tunable HSV gate) and tracks movement</li>
            <li>Per-fish <b>speed</b> and <b>dwell-zone</b> logic to flag likely fatalities</li>
            <li>Configurable thresholds, evidence snapshots, and live insights</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown("### ")
    st.markdown(
        """
        <div class="card">
          <b>Quick tips</b><br>
          • Upload a short 10–20s clip for fast iteration<br>
          • Tune stillness window & speed threshold per tank<br>
          • Snapshots saved under <code>alerts/</code>
        </div>
        """,
        unsafe_allow_html=True
    )

metrics_strip = st.empty()
metrics_strip.markdown(
    '<div class="pill">Frame: —×—</div><div class="pill">FPS: —</div><div class="pill">Tracks: —</div>',
    unsafe_allow_html=True
)

with st.expander("How it works", expanded=False):
    st.markdown(
        """
        1) Apply an HSV color gate to isolate orange fish bodies (sidebar).  
        2) Track centroid motion over time with a lightweight ID tracker.  
        3) Compute average velocity over a sliding window; if **low** and the fish dwells near the **top** or **bottom** zones, raise a flag.  
        4) Store cropped snapshots for review and audit.
        """
    )

# =========================
# Sidebar — Controls
# =========================
with st.sidebar:
    st.header("Video Source")
    source_mode = st.radio("Choose source", ["Upload MP4", "Webcam (0)", "RTSP URL"], index=0)
    uploaded_file = None
    rtsp_url = ""
    if source_mode == "Upload MP4":
        uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mpeg", "m4v"])
    elif source_mode == "RTSP URL":
        rtsp_url = st.text_input("RTSP URL", value="", help="e.g., rtsp://user:pass@ip/stream")

    st.markdown("---")
    st.header("Color Gate (HSV)")
    H_low   = st.slider("H low", 0, 179, 5)
    S_low   = st.slider("S low", 0, 255, 120)
    V_low   = st.slider("V low", 0, 255, 120)
    H_high  = st.slider("H high", 0, 179, 24)
    S_high  = st.slider("S high", 0, 255, 255)
    V_high  = st.slider("V high", 0, 255, 255)

    st.markdown("---")
    st.header("Decision Rules")
    window_sec   = st.slider("Stillness window (seconds)", 3, 60, 12)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 200, 25)
    top_zone_pct = st.slider("Top zone height (%)", 2, 40, 12)
    bot_zone_pct = st.slider("Bottom zone height (%)", 2, 40, 12)

    st.markdown("---")
    show_mask = st.checkbox("Show mask overlay", value=False)

    st.markdown("---")
    st.header("Run")
    start = st.button("▶️ Process")
    stop  = st.button("⏹ Stop")
    if "running" not in st.session_state:
        st.session_state.running = False
    if start: st.session_state.running = True
    if stop:  st.session_state.running = False

# Placeholders
preview       = st.empty()
insights_box  = st.container()
status_banner = st.empty()

# =========================
# Simple centroid tracker
# =========================
@dataclass
class Track:
    tid: int
    history: deque = field(default_factory=lambda: deque(maxlen=256))  # (t, x, y)
    last_bbox: tuple = (0, 0, 0, 0)  # x1, y1, x2, y2
    last_seen: float = 0.0
    last_alert_t: float = 0.0

class CentroidTracker:
    def __init__(self, dist_thresh=50, max_age=30):
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.dist_thresh = dist_thresh
        self.max_age = max_age  # frames

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, centroids, bboxes, tstamp):
        """centroids: [(cx,cy), ...]; bboxes: [(x1,y1,x2,y2), ...]"""
        # Age out old tracks
        for tr in list(self.tracks.values()):
            tr.last_seen += 1
            if tr.last_seen > self.max_age:
                del self.tracks[tr.tid]

        # Match by nearest
        assigned = set()
        for i, c in enumerate(centroids):
            best_id, best_d = None, 1e9
            for tr in self.tracks.values():
                d = self._dist(c, tr.history[-1][1:3]) if tr.history else 1e9
                if d < best_d and d <= self.dist_thresh:
                    best_id, best_d = tr.tid, d
            if best_id is not None:
                tr = self.tracks[best_id]
                tr.history.append((tstamp, c[0], c[1]))
                tr.last_bbox = bboxes[i]
                tr.last_seen = 0
                assigned.add(i)
            else:
                # New track
                tid = self.next_id; self.next_id += 1
                tr = Track(tid=tid)
                tr.history.append((tstamp, c[0], c[1]))
                tr.last_bbox = bboxes[i]
                tr.last_seen = 0
                self.tracks[tid] = tr

        return self.tracks

# =========================
# Helpers
# =========================
def open_capture(mode, rtsp, upload):
    if mode == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    elif mode == "RTSP URL":
        cap = cv2.VideoCapture(rtsp)
    else:
        if upload is None:
            return None
        tmp = "tmp_upload.mp4"
        with open(tmp, "wb") as f:
            f.write(upload.read())
        cap = cv2.VideoCapture(tmp)
    if not cap or not cap.isOpened():
        return None
    return cap

def hsv_detect(frame, hsv_range):
    H_low, S_low, V_low, H_high, S_high, V_high = hsv_range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (H_low, S_low, V_low), (H_high, S_high, V_high))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids, bboxes = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 80:  # ignore tiny
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w/2, y + h/2
        centroids.append((cx, cy))
        bboxes.append((x, y, x+w, y+h))
    return centroids, bboxes, mask

def avg_speed_px_per_s(track: Track, now_t: float, window_sec: float):
    # average over last window_sec
    pts = [p for p in track.history if now_t - p[0] <= window_sec]
    if len(pts) < 2:
        return 0.0
    dist = 0.0
    dt = pts[-1][0] - pts[0][0]
    for i in range(1, len(pts)):
        dist += math.hypot(pts[i][1]-pts[i-1][1], pts[i][2]-pts[i-1][2])
    if dt <= 1e-3:
        return 0.0
    return dist / dt  # px/sec

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def summarize_insights(flagged, moving, silent, top_cnt, bottom_cnt):
    if flagged == 0 and moving > 0:
        return f"All {moving} tracked fish appear active and healthy."
    if flagged > 0:
        parts = []
        if top_cnt:
            parts.append(f"{top_cnt} near the surface")
        if bottom_cnt:
            parts.append(f"{bottom_cnt} near the bottom")
        zone_txt = (", ".join(parts)) if parts else "alert zone(s)"
        return (f"{flagged} fish are likely dead or distressed "
                f"(low motion in {zone_txt}).")
    if moving == 0 and silent > 0:
        return "Fish detected but insufficient motion history yet."
    return "No fish detected."

# =========================
# Main loop
# =========================
alerts_dir = "alerts"; ensure_dir(alerts_dir)
cap = open_capture(source_mode, rtsp_url, uploaded_file)

if not cap:
    status_banner.warning("Provide a valid video source (upload a video, use Webcam 0, or set an RTSP URL).")
    st.stop()

tracker = CentroidTracker(dist_thresh=60, max_age=15)
fps_smooth, last_t = 0.0, time.time()

event_log = deque(maxlen=12)  # keep a small rolling log for display

while st.session_state.running:
    ok, frame = cap.read()
    if not ok:
        status_banner.info("End of stream or cannot read more frames.")
        break

    now_t = time.time()
    dt = now_t - last_t
    last_t = now_t
    fps = 1.0 / dt if dt > 1e-3 else 0.0
    fps_smooth = fps if fps_smooth == 0 else (0.9*fps_smooth + 0.1*fps)

    h, w = frame.shape[:2]
    top_h = int(h * (top_zone_pct/100.0))
    bot_h = int(h * (bot_zone_pct/100.0))

    # Detect via HSV
    centroids, bboxes, mask = hsv_detect(frame, (H_low, S_low, V_low, H_high, S_high, V_high))

    # Track
    tracks = tracker.update(centroids, bboxes, now_t)

    # Draw
    overlay = frame.copy()
    # Zones
    cv2.rectangle(overlay, (0,0), (w, top_h), (255, 0, 0), 2)
    cv2.rectangle(overlay, (0,h-bot_h), (w, h), (0, 0, 255), 2)

    flagged = 0; moving = 0; silent = 0; top_cnt = 0; bottom_cnt = 0
    candidates = []

    for tr in tracks.values():
        if not tr.history:
            continue
        cx, cy = int(tr.history[-1][1]), int(tr.history[-1][2])
        x1,y1,x2,y2 = map(int, tr.last_bbox)

        # avg speed
        v = avg_speed_px_per_s(tr, now_t, window_sec)
        at_top = cy <= top_h
        at_bottom = cy >= (h - bot_h)
        likely = (v < speed_thresh) and (at_top or at_bottom)

        if v >= speed_thresh:
            moving += 1
        else:
            silent += 1
        if at_top: top_cnt += 1
        if at_bottom: bottom_cnt += 1
        if likely: flagged += 1

        color = (0,255,0)
        label = f"id={tr.tid} v={v:.0f}px/s"
        if likely:
            color = (0,0,255)
            label += "  ⚠ likely dead"
            if (now_t - tr.last_alert_t) > 10.0:
                tr.last_alert_t = now_t
                crop = overlay[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = os.path.join(alerts_dir, f"alert_{ts}_id{tr.tid}.jpg")
                if crop.size > 0:
                    cv2.imwrite(out_path, crop)
                    candidates.append(out_path)
                    event_log.appendleft(f"{ts}: Track {tr.tid} flagged (v={v:.0f}px/s).")

        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        cv2.putText(overlay, label, (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        # trajectory (last 20)
        pts = list(tr.history)[-20:]
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][1]), int(pts[i-1][2]))
            p2 = (int(pts[i][1]),   int(pts[i][2]))
            cv2.line(overlay, p1, p2, (200,200,0), 1)

    # Mask toggle
    if show_mask:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 0.8, mask_vis, 0.4, 0)

    # UI updates
    metrics_strip.markdown(
        f'<div class="pill">Frame: {w}×{h}</div>'
        f'<div class="pill">FPS: {fps_smooth:.1f}</div>'
        f'<div class="pill">Tracks: {len(tracks)}</div>',
        unsafe_allow_html=True
    )
    preview.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Insights panel
    summary = summarize_insights(flagged, moving, silent, top_cnt, bottom_cnt)
    with insights_box:
        st.subheader("Insights")
        st.write(summary)

        cols = st.columns(4)
        cols[0].metric("Tracked fish", f"{len(tracks)}")
        cols[1].metric("Likely dead", f"{flagged}")
        cols[2].metric("Avg window (s)", f"{window_sec}")
        cols[3].metric("Speed threshold", f"{speed_thresh} px/s")

        if candidates:
            st.markdown("**Recent alert snapshots**")
            ccols = st.columns(min(4, len(candidates)))
            for i, p in enumerate(candidates[:4]):
                ccols[i % len(ccols)].image(p, use_column_width=True)

        if event_log:
            st.markdown("**Event log**")
            for e in list(event_log)[:6]:
                st.write(f"• {e}")

    time.sleep(0.01)

st.success("Done. Open the 'alerts' folder to see any snapshots.")
try:
    cap.release()
except Exception:
    pass
