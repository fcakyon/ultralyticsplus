# ultralytics+

Extra features for [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).

## installation

```bash
pip install ultralyticsplus
```

## push to 🤗 hub

```bash
ultralyticsplus --exp_dir runs/detect/train --hf_model_id HF_USERNAME/MODELNAME
```

## load from 🤗 hub

```python
from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('HF_USERNAME/MODELNAME')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model.predict(image, imgsz=640)

# parse results
result = results[0]
boxes = result.boxes.xyxy # x1, y1, x2, y2
scores = result.boxes.conf
categories = result.boxes.cls
scores = result.probs # for classification models
masks = result.masks # for segmentation models

# show results on image
render = render_result(model=model, image=image, result=result)
render.show()
```
