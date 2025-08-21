# classic_detector.py
import cv2, numpy as np

class ClassicDetector:
    def __init__(self, min_area=900, lower_hsv=(5,100,80), upper_hsv=(25,255,255), orange_ratio=0.20):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.min_area = min_area
        self.lower = np.array(lower_hsv, np.uint8)
        self.upper = np.array(upper_hsv, np.uint8)
        self.orange_ratio = orange_ratio

    def _orange_ratio(self, roi):
        if roi.size == 0: return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def infer(self, frame):
        fg = self.bg.apply(frame, learningRate=0.002)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
        fg = cv2.dilate(fg, np.ones((3,3),np.uint8), iterations=2)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H,W = frame.shape[:2]
        out = []
        for c in cnts:
            if cv2.contourArea(c) < self.min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            roi = frame[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
            if self._orange_ratio(roi) < self.orange_ratio:
                continue
            cx, cy = x+w/2.0, y+h/2.0
            out.append({"tid": None, "cls_id": 0, "cls_name": "goldfish", "conf": 0.99,
                        "xyxy": (float(x),float(y),float(x+w),float(y+h)), "center": (cx,cy)})
        return out
