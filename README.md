# Dead Goldfish Detector — YOLO + ByteTrack Scaffold

A production-leaning scaffold that uses **Ultralytics YOLO** for detection and **ByteTrack** for multi-object tracking.
We then apply a simple **stillness + top/bottom dwell** rule to flag **likely dead** goldfish.

## 📦 Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app_yolo.py
```

In the sidebar, set:
- **Model path** → your trained weights (e.g., `runs/detect/train/weights/best.pt`)
- **Goldfish class keywords** → e.g., `goldfish` (the app will treat any class name containing these keywords as goldfish)
- Choose a **Video Source** (webcam, RTSP, or upload an `.mp4`)

## 🧑‍🏫 How to Train a Goldfish Model (Ultralytics)

1. **Label data** (goldfish, optionally other_fish, debris). Use tools like Label Studio, CVAT, or Roboflow.
2. Prepare a dataset YAML (see `data/goldfish_dataset.yaml`), for example:

```yaml
# data/goldfish_dataset.yaml
path: ../datasets/goldfish   # <- root folder containing images/ and labels/
train: images/train
val: images/val
test: images/test

names:
  0: goldfish
  1: other_fish
  2: debris
```

3. **Train**:
```bash
yolo detect train data=data/goldfish_dataset.yaml model=yolov8n.pt epochs=80 imgsz=640 batch=16
# After training, weights are in runs/detect/train/weights/best.pt
```

4. **Export** (optional):
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

## 🧠 How the App Works

- `yolo_infer.py`: wraps `ultralytics.YOLO` and calls `model.track(..., persist=True, tracker='bytetrack.yaml')` each frame.
- `track_store.py`: stores short motion histories per YOLO track ID.
- `decision.py`: computes per-track average speed and applies the stillness + top/bottom dwell rule.
- `app_yolo.py`: Streamlit UI + video capture + drawing + alert snapshot saving.

## ⚙️ Tips

- If your lighting varies, start with a lower **confidence** (0.25) and increase as you improve the dataset.
- For crowded tanks / occlusions, try `botsort.yaml` instead of `bytetrack.yaml`.
- To reduce false positives on sleeping fish, lengthen **Stillness window** or lower **speed threshold**.
- Add a simple **Confirm/Reject** review step to collect hard negatives and retrain.

## 🚀 Streamlit Cloud

- Push this folder to a public GitHub repo, then deploy on Streamlit Community Cloud.
- Add any RTSP credentials via **Secrets**, never commit them to the repo.

## 📁 Repo Layout

```
.
├─ app_yolo.py
├─ yolo_infer.py
├─ track_store.py
├─ decision.py
├─ requirements.txt
├─ README.md
└─ data/
   └─ goldfish_dataset.yaml  # template; fill in your paths
```

## 🔄 Next Steps

- Add a **species classifier** (when you expand beyond goldfish).
- Integrate **active learning**: store every alert + human decision; use it to retrain.
- Consider **Eulerian Video Magnification** on gill area for borderline cases (compute-heavy but useful).
