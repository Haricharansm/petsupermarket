# yolo_infer.py — wrapper around Ultralytics YOLO detect+track
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Ultralytics not found. Install with `pip install ultralytics`. "
        "If deploying on Streamlit Cloud, ensure it is listed in requirements.txt."
    ) from e

class YOLOTracker:
    def __init__(self, model_path: str, conf: float=0.25, iou: float=0.5, imgsz: int=640, tracker: str="bytetrack.yaml", device: Optional[str]=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        # Use Ultralytics built-in tracking with persist=True
        res = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker,
            device=self.device,
            verbose=False
        )
        results = []
        if not res:
            return results

        r = res[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return results

        # Safely extract tensors
        xyxy = boxes.xyxy
        ids = boxes.id
        clses = boxes.cls
        confs = boxes.conf

        # Class names mapping
        names = getattr(self.model, "names", {})
        def name_of(idx: int):
            if isinstance(names, dict):
                return names.get(idx, str(idx))
            elif isinstance(names, (list, tuple)) and idx >= 0 and idx < len(names):
                return names[idx]
            return str(idx)

        # Convert to CPU numpy
        xyxy = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else xyxy
        ids_np = ids.detach().cpu().numpy() if ids is not None and hasattr(ids, "detach") else None
        clses_np = clses.detach().cpu().numpy() if clses is not None and hasattr(clses, "detach") else None
        confs_np = confs.detach().cpu().numpy() if confs is not None and hasattr(confs, "detach") else None

        n = xyxy.shape[0] if xyxy is not None else 0
        for i in range(n):
            x1,y1,x2,y2 = xyxy[i]
            tid = int(ids_np[i]) if ids_np is not None and ids_np[i] is not None else None
            cls_id = int(clses_np[i]) if clses_np is not None else -1
            conf = float(confs_np[i]) if confs_np is not None else None
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            results.append({
                "tid": tid,
                "cls_id": cls_id,
                "cls_name": name_of(cls_id),
                "conf": conf,
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "center": (cx, cy)
            })
        return results
