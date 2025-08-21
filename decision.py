# decision.py — works for both the modular and YOLO scaffolds (no imports needed)
import math

def compute_avg_speed(track, now_t: float, window_s: float) -> float:
    """
    track.history is a deque of (t, x, y).
    Returns average speed in px/sec over the last `window_s` seconds.
    """
    pts = [p for p in getattr(track, "history", []) if now_t - p[0] <= window_s]
    if len(pts) < 2:
        return 0.0

    dist = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][1] - pts[i-1][1]
        dy = pts[i][2] - pts[i-1][2]
        dist += math.hypot(dx, dy)

    total_time = max(1e-3, pts[-1][0] - pts[0][0])
    return dist / total_time  # px/sec

def is_likely_dead(avg_speed: float, at_top: bool, at_bottom: bool, speed_thresh: float) -> bool:
    return (avg_speed <= speed_thresh) and (at_top or at_bottom)
