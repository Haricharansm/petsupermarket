# Demo Goldfish Detection Dataset (Synthetic)

This is a small, synthetic dataset for quick demos. It contains orange **goldfish** ellipses, blue **other_fish**, and gray **debris** rendered over water-like backgrounds.

## Structure
```
demo_goldfish_dataset/
├─ images/{train,val,test}/*.jpg
├─ labels/{train,val,test}/*.txt  # YOLO format
└─ data.yaml
```

## Classes
- `0` goldfish
- `1` other_fish
- `2` debris

## Train (Ultralytics)
```bash
yolo detect train data=demo_goldfish_dataset/data.yaml model=yolov8n.pt epochs=30 imgsz=640 batch=16
```
