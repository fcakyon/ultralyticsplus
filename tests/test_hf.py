from ultralyticsplus import (
    YOLO,
    download_from_hub,
    postprocess_classify_output,
    render_result,
    push_to_hfhub,
)
import subprocess
import os

hub_id = "ultralyticsplus/yolov8s"


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


def run(cmd):
    # Run a subprocess command with check=True
    subprocess.run(cmd.split(), check=True)


def test_detection_upload():
    import platform
    from packaging.version import Version

    # run following lines if linux and python major == 3 and python minor == 10 (python micor can be anything)
    print(f'platform.system(): {platform.system()}')
    print(f'platform.python_version(): {platform.python_version()}')
    print(f'os.getenv("RUNNER_OS"): {os.getenv("RUNNER_OS")}')
    print(f'os.getenv("MATRIX_OS"): {os.getenv("MATRIX_OS")}')
    print(f'Version(platform.python_version()) >= Version("3.10"): {Version(platform.python_version()) >= Version("3.10")}')
    if os.getenv('MATRIX_OS') == 'ubuntu-latest' and Version(platform.python_version()) >= Version("3.10"):
        run('yolo train detect model=yolov8n.pt data=coco8.yaml imgsz=32 epochs=1')
        hf_token = os.getenv('HF_TOKEN')
        if hf_token is None:
            raise ValueError('Please set HF_TOKEN environment variable to your HuggingFace token.')
        push_to_hfhub(
            hf_model_id="fcakyon/yolov8n-test",
            exp_dir='runs/detect/train',
            hf_token=os.getenv('HF_TOKEN'),
            hf_private=True,
            hf_dataset_id="fcakyon/football-detection",
            thumbnail_text='YOLOv8s Football Detection'
        )