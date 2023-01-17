from ultralyticsplus import (
    YOLO,
    download_from_hub,
    postprocess_classify_output,
    render_result,
)

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
    image = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    # perform inference
    results = model(image, imgsz=640)
    render = render_result(model=model, image=image, result=results[0])
    render.show()


def test_segmentation_inference():
    model = YOLO("yolov8n-seg.pt")

    # set image
    image = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    # perform inference
    results = model(image, imgsz=640)
    render = render_result(model=model, image=image, result=results[0])
    render.show()


def test_classification_inference():
    model = YOLO("yolov8n-cls.pt")

    # set image
    image = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    # perform inference
    results = model(image, imgsz=640)
    name_to_probs = postprocess_classify_output(model=model, result=results[0])
    name_to_probs


def test_inference_generic():
    model = YOLO(hub_id_generic)

    # set image
    image = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    # perform inference
    result = model.predict(image, imgsz=640)
    render = render_result(model=model, image=image, result=result)
    render.show()
