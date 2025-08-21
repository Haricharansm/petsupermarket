# save as tools/mine_frames.py
import cv2, os, time
import numpy as np

def mine(input_mp4, out_dir="mined_frames", stride=5, top_pct=0.12, bot_pct=0.12, area_min=900):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_mp4)
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    idx = 0; saved = 0
    ok, frame = cap.read()
    if not ok: return
    h,w = frame.shape[:2]
    top_h, bot_h = int(h*top_pct), int(h*bot_pct)
    while ok:
        if idx % stride == 0:
            fg = bg.apply(frame, learningRate=0.002)
            fg = cv2.medianBlur(fg, 5)
            cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hit = False
            for c in cnts:
                if cv2.contourArea(c) < area_min: continue
                x,y,bw,bh = cv2.boundingRect(c)
                cy = y + bh/2
                if cy <= top_h or cy >= (h-bot_h):
                    hit = True; break
            if hit or idx % (stride*50) == 0:  # also sample periodic normals
                fp = os.path.join(out_dir, f"frame_{idx:07d}.jpg")
                cv2.imwrite(fp, frame); saved += 1
        ok, frame = cap.read(); idx += 1
    print("saved:", saved)

if __name__ == "__main__":
    import sys; mine(sys.argv[1])
