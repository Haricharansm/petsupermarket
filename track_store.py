# track_store.py — track history and metadata keyed by YOLO track IDs
from dataclasses import dataclass, field
from typing import Dict, Tuple, Deque, List
from collections import deque

@dataclass
class Track:
    tid: int
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=6000))  # (t, x, y)
    last_alert_t: float = 0.0
    cls_name: str = "unknown"

class TrackStore:
    def __init__(self, maxlen: int = 6000, stale_sec: float = 12.0):
        self.tracks: Dict[int, Track] = {}
        self.maxlen = maxlen
        self.stale_sec = stale_sec

    def update_from_dets(self, dets: List[dict], now_t: float):
        seen = set()
        for d in dets:
            tid = d.get("tid", None)
            if tid is None:
                continue
            seen.add(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = Track(tid=tid)
                self.tracks[tid] = tr
            cx, cy = d["center"]
            tr.history.append((now_t, cx, cy))
            tr.cls_name = d.get("cls_name", tr.cls_name)

        # prune stale tracks
        stale = []
        for tid, tr in self.tracks.items():
            if not tr.history:
                continue
            last_t = tr.history[-1][0]
            if (now_t - last_t) > self.stale_sec:
                stale.append(tid)
        for tid in stale:
            self.tracks.pop(tid, None)
