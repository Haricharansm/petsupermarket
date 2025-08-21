# save as tools/dets_to_yolo.py
# expects a list of detections per image: [(cls, x1,y1,x2,y2), ...]
import os, json

def to_line(cls_id, x1,y1,x2,y2, w,h):
    cx = (x1+x2)/2/w; cy=(y1+y2)/2/h; bw=(x2-x1)/w; bh=(y2-y1)/h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def write_labels(det_json, labels_dir, names):
    os.makedirs(labels_dir, exist_ok=True)
    data = json.load(open(det_json))
    for im, dets in data.items():
        h,w = dets.get("hw", [0,0])
        lines = []
        for cls_name, x1,y1,x2,y2 in dets["boxes"]:
            cls_id = names.index(cls_name)
            lines.append(to_line(cls_id, x1,y1,x2,y2, w,h))
        with open(os.path.join(labels_dir, os.path.splitext(os.path.basename(im))[0]+".txt"), "w") as f:
            f.write("\n".join(lines))
