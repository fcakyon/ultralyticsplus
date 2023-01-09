# YOLOv8 to 🤗
HuggingFace utilities for Ultralytics/YOLOv8

## installation

```bash
pip install yolov8tohf
```

## push to hub

```bash
yolov8tohf --exp_dir runs/detect/train --hf_model_id HF_USERNAME/MODELNAME
```

## load from hub

```python
from yolov8tohf import YOLO

# load model
model = YOLO('fcakyon/yolov8s-test')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
model.predict(img, imgsz=640)
```