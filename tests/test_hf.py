from ultralyticsplus import YOLO, render_predictions, download_from_hub

hub_id = "ultralyticsplus/yolov8s"
hub_id_generic = "kadirnar/yolov8n-v8.0"


def test_load_from_hub():
    path = download_from_hub(hub_id)


def test_yolo_from_hub():
    model = YOLO(hub_id)

    # set model parameters
    model.overrides["conf"] = 0.25  # NMS confidence threshold
    model.overrides["iou"] = 0.45  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 1000  # maximum number of detections per image


def test_inference():
    model = YOLO(hub_id)

    # set image
    img = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    # perform inference
    for result in model.predict(img, imgsz=640, return_outputs=True):
        print(result)  # [x1, y1, x2, y2, conf, class]
        render = render_predictions(model, img=img, det=result["det"])
        render.show()


def test_inference_generic():
    model = YOLO(hub_id_generic)

    # set image
    img = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    # perform inference
    for result in model.predict(img, imgsz=640, return_outputs=True):
        print(result)  # [x1, y1, x2, y2, conf, class]
        render = render_predictions(model, img=img, det=result["det"])
        render.show()
