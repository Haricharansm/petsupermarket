# app_classic_demo.py
# Streamlit demo WITHOUT training: color + motion + simple tracking
import os, time, math, uuid
from collections import deque
from typing import Dict, Deque, Tuple, List

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Dead Goldfish Detector (Classic Demo)", layout="wide")
st.title("🐟 Dead Goldfish Detector — Classic (no training)")

st.markdown("""
This quick demo uses **color + motion** and a **simple tracker** to flag *likely dead* goldfish:

- Detect **orange** fish via HSV color mask  
- Track centroids across frames  
- If **avg speed** is low for *N* seconds **and** the fish dwells near the **top** or **bottom**, raise an alert  

> Perfect for a live demo while your YOLO model is still training.
""")

# ------------------------
# Sidebar controls
# ------------------------
with st.sidebar:
    st.header("Video Source")
    source_mode = st.radio("Choose source", ["Upload MP4", "Webcam (0)"], index=0)
    uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi"])
    st.caption("Tip: small 10–20s clips run fastest.")

    st.header("Color Gate (HSV)")
    st.write("Default targets orange goldfish; adjust if needed")
    h_low  = st.slider("H low", 0, 179, 5)
    h_high = st.slider("H high", 0, 179, 30)
    s_low  = st.slider("S low", 0, 255, 120)
    s_high = st.slider("S high", 0, 255, 255)
    v_low  = st.slider("V low", 0, 255, 90)
    v_high = st.slider("V high", 0, 255, 255)
    min_area = st.slider("Min blob area (px)", 50, 5000, 600, 50)
    blur_k   = st.slider("Blur (odd kernel)", 1, 15, 5, 2)

    st.header("Decision Rules")
    window_sec   = st.slider("Stillness window (seconds)", 3, 60, 12)
    speed_thresh = st.slider("Avg speed threshold (px/sec)", 1, 200, 25)
    top_pct      = st.slider("Top zone height (%)", 2, 40, 12)
    bottom_pct   = st.slider("Bottom zone height (%)", 2, 40, 12)

    st.header("Run")
    run_btn  = st.button("▶️ Process")
    stop_btn = st.button("⏹ Stop")
    if "running" not in st.session_state: st.session_state.running = False
    if run_btn:  st.session_state.running = True
    if stop_btn: st.session_state.running = False

# ------------------------
# Helpers
# ------------------------
def open_capture():
    if source_mode == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    else:
        if uploaded_file is None:
            return None
        tmp_path = "classic_tmp.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp_path)
    if not cap or not cap.isOpened():
        return None
    return cap

def orange_mask(bgr, hsv_lo, hsv_hi, blur_k=5):
    img = bgr
    if blur_k >= 3 and blur_k % 2 == 1:
        img = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8), iterations=1)
    return mask

def detect_candidates(frame, hsv_lo, hsv_hi, min_area, blur_k):
    mask = orange_mask(frame, hsv_lo, hsv_hi, blur_k)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, centers = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: 
            continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,x+w,y+h))
        centers.append((x + w/2.0, y + h/2.0))
    return boxes, centers, mask

# very simple nearest-neighbor tracker
class Track:
    def __init__(self, cx, cy, t):
        self.id = str(uuid.uuid4())[:8]
        self.history: Deque[Tuple[float,float,float]] = deque(maxlen=6000)  # (t,x,y)
        self.last_t = t
        self.last_xy = (cx, cy)
        self.last_alert_t = 0.0
        self.history.append((t, cx, cy))

    def update(self, cx, cy, t):
        self.last_t = t
        self.last_xy = (cx, cy)
        self.history.append((t, cx, cy))

Tracks: Dict[str, Track] = {}
def step_tracker(detections: List[Tuple[float,float]], t: float, max_dist=60.0, stale_sec=4.0):
    global Tracks
    # match by nearest distance
    dets = detections[:]
    used = set()
    # try to match existing tracks
    for tid, tr in list(Tracks.items()):
        if not dets: break
        dists = [math.hypot(tr.last_xy[0]-cx, tr.last_xy[1]-cy) for (cx,cy) in dets]
        j = int(np.argmin(dists)) if dists else -1
        if j >= 0 and dists[j] <= max_dist:
            cx,cy = dets[j]
            tr.update(cx,cy,t)
            used.add(j)
        # drop stale tracks
        if t - tr.last_t > stale_sec:
            del Tracks[tid]

    # new tracks for unmatched detections
    for j,(cx,cy) in enumerate(dets):
        if j in used: 
            continue
        tr = Track(cx,cy,t)
        Tracks[tr.id] = tr

def avg_speed_px_sec(tr: Track, now_t: float, win_sec: float) -> float:
    pts = list(tr.history)
    if len(pts) < 2:
        return 0.0
    # use last win_sec seconds
    tail = [p for p in pts if now_t - p[0] <= win_sec]
    if len(tail) < 2:
        tail = pts[-min(10, len(pts)):]  # small fallback
    dist = 0.0
    dt    = tail[-1][0] - tail[0][0]
    for i in range(1, len(tail)):
        _,x1,y1 = tail[i-1]
        _,x2,y2 = tail[i]
        dist += math.hypot(x2-x1, y2-y1)
    if dt <= 1e-6:
        return 0.0
    return dist / dt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ------------------------
# Main
# ------------------------
alerts_dir = "alerts"
ensure_dir(alerts_dir)
frame_info = st.empty()
preview   = st.empty()
mask_view = st.checkbox("Show mask", value=False)

cap = open_capture()
if not cap:
    st.warning("Provide a valid source (upload a video or use Webcam 0).")
    st.stop()

fps_smooth = 0.0
last_t = time.time()

while st.session_state.running:
    ok, frame = cap.read()
    if not ok:
        st.info("End of stream or cannot read more frames.")
        break

    now_t = time.time()
    dt = now_t - last_t
    last_t = now_t
    fps = 1.0/dt if dt>1e-3 else 0.0
    fps_smooth = fps if fps_smooth == 0 else 0.9*fps_smooth + 0.1*fps

    h,w = frame.shape[:2]
    top_h = int(h * (top_pct/100.0))
    bot_h = int(h * (bottom_pct/100.0))

    hsv_lo = np.array([h_low, s_low, v_low], dtype=np.uint8)
    hsv_hi = np.array([h_high, s_high, v_high], dtype=np.uint8)

    boxes, centers, mask = detect_candidates(frame, hsv_lo, hsv_hi, min_area, blur_k)
    step_tracker(centers, now_t)

    overlay = frame.copy()
    # zones
    cv2.rectangle(overlay, (0,0), (w, top_h), (255,0,0), 2)
    cv2.rectangle(overlay, (0,h-bot_h), (w, h), (0,0,255), 2)

    # draw detections
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

    # decisions per track
    alerts = 0
    for tid,tr in list(Tracks.items()):
        cx,cy = tr.last_xy
        avg_v  = avg_speed_px_sec(tr, now_t, window_sec)
        at_top = cy <= top_h
        at_bot = cy >= (h - bot_h)
        likely = (avg_v < speed_thresh) and (at_top or at_bot)

        color = (0,255,0)
        label = f"id={tid} v={avg_v:.0f}px/s"
        if likely and (now_t - tr.last_alert_t) > 5.0:
            tr.last_alert_t = now_t
            color = (0,0,255)
            label += "  ⚠ likely dead"
            # snapshot
            sx1,sy1 = int(max(0,cx-60)), int(max(0,cy-40))
            sx2,sy2 = int(min(w,cx+60)), int(min(h,cy+40))
            crop = frame[sy1:sy2, sx1:sx2]
            ts = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(os.path.join(alerts_dir, f"alert_{ts}_id{tid}.jpg"), crop)
            alerts += 1

        cv2.circle(overlay, (int(cx), int(cy)), 6, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, label, (int(cx)+8, int(cy)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # draw small trail
        pts = list(tr.history)[-20:]
        for i in range(1, len(pts)):
            _,x1,y1 = pts[i-1]
            _,x2,y2 = pts[i]
            cv2.line(overlay, (int(x1),int(y1)), (int(x2),int(y2)), (200,200,0), 1)

    if mask_view:
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = np.hstack([overlay, m3])

    frame_info.write(f"Frame: {w}×{h} | FPS: {fps_smooth:.1f} | Tracks: {len(Tracks)}")
    preview.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # small sleep to keep Streamlit responsive
    time.sleep(0.005)

try:
    cap.release()
except Exception:
    pass

st.success("Done. Open the 'alerts' folder to see any snapshots.")
