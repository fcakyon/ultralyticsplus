import logging
import os
from pathlib import Path

import pandas as pd

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)
LOGGER = logging.getLogger(__name__)


def generate_model_usage_markdown(
    repo_id, ap50, task="object-detection", input_size=640, dataset_id=None, labels=None
):
    from ultralytics import __version__ as ultralytics_version

    if dataset_id is not None:
        datasets_str_1 = f"""
datasets:
- {dataset_id}
"""
        datasets_str_2 = f"""
    dataset:
      type: {dataset_id}
      name: {dataset_id}
      split: validation
"""
    else:
        datasets_str_1 = datasets_str_2 = ""
    return f""" 
---
tags:
- yolov8tohf
- yolov8
- ultralytics
- yolo
- vision
- {task}
- pytorch
library_name: ultralytics
library_version: {ultralytics_version}
inference: false
{datasets_str_1}
model-index:
- name: {repo_id}
  results:
  - task:
      type: {task}
{datasets_str_2}
    metrics:
      - type: precision  # since mAP@0.5 is not available on hf.co/metrics
        value: {ap50}  # min: 0.0 - max: 1.0
        name: mAP@0.5
---

<div align="center">
  <img width="640" alt="{repo_id}" src="https://huggingface.co/{repo_id}/resolve/main/sample_visuals.jpg">
</div>

### Supported Labels

```
{labels}
```

### How to use

- Install `ultralytics` and `yolov8tohf`:

```bash
pip install -U ultralytics yolov8tohf
```

- Load model and perform prediction:

```python
from yolov8tohf import YOLO

# load model
model = YOLO('{repo_id}')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
model.predict(img, imgsz={input_size})
```

"""


def push_model_card_to_hfhub(
    repo_id,
    exp_folder,
    ap50,
    hf_token=None,
    input_size=640,
    task="object-detection",
    private=False,
    dataset_id=None,
):
    from huggingface_hub import upload_file, create_repo
    from ultralytics import YOLO

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )

    # upload sample visual to the repo
    sample_visual_path = Path(exp_folder) / "val_batch0_labels.jpg"
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(sample_visual_path),
        path_in_repo="sample_visuals.jpg",
        commit_message="upload sample visuals",
        token=hf_token,
        repo_type="model",
    )

    # Create model card
    best_model_path = Path(exp_folder) / "weights/best.pt"
    model = YOLO(model=best_model_path)
    labels = list(model.model.names.values())
    modelcard_markdown = generate_model_usage_markdown(
        repo_id,
        task=task,
        input_size=input_size,
        dataset_id=dataset_id,
        ap50=ap50,
        labels=labels,
    )
    modelcard_path = Path(exp_folder) / "README.md"
    with open(modelcard_path, "w") as file_object:
        file_object.write(modelcard_markdown)
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(modelcard_path),
        path_in_repo=Path(modelcard_path).name,
        commit_message="Add yolov5 model card",
        token=hf_token,
        repo_type="model",
    )


def push_config_to_hfhub(
    repo_id,
    exp_folder,
    best_ap50=None,
    input_size=640,
    task="object-detection",
    hf_token=None,
    private=False,
):
    """
    Pushes a yolov5 config to huggingface hub

    Arguments:
        repo_id (str): The name of the repository to create on huggingface.co
        exp_folder (str): The path to the experiment folder
        best_ap50 (float): The best ap50 score of the model
        input_size (int): The input size of the model (default: 640)
        task (str): The task of the model (default: object-detection)
        hf_token (str): The huggingface token to use to push the model
        private (bool): Whether the model should be private or not
    """
    from huggingface_hub import upload_file, create_repo
    import json

    config = {"input_size": input_size, "task": task, "best_ap50": best_ap50}
    config_path = Path(exp_folder) / "config.json"
    with open(config_path, "w") as file_object:
        json.dump(config, file_object)

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(config_path),
        path_in_repo=Path(config_path).name,
        commit_message="Add yolov5 config",
        token=hf_token,
        repo_type="model",
    )


def push_model_to_hfhub(repo_id, exp_folder, hf_token=None, private=False):
    """
    Pushes a yolov5 model to huggingface hub

    Arguments:
        repo_id (str): huggingface repo id to be uploaded to
        exp_folder (str): yolov5 experiment folder path
        hf_token (str): huggingface write token
        private (bool): whether to make the repo private or not
    """
    from huggingface_hub import upload_file, create_repo, list_repo_files, delete_file
    from glob import glob

    best_model_path = Path(exp_folder) / "weights/best.pt"

    # remove present tensorboard log from huggingface hub repo
    for file in list_repo_files(repo_id, token=hf_token):
        if file.startswith("events.out.tfevents"):
            delete_file(path_in_repo=file, repo_id=repo_id, token=hf_token)

    tensorboard_log_path = glob(f"{exp_folder}/events.out.tfevents*")[-1]

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(best_model_path),
        path_in_repo=Path(best_model_path).name,
        commit_message="Upload yolov8 best model",
        token=hf_token,
        repo_type="model",
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(tensorboard_log_path),
        path_in_repo=Path(tensorboard_log_path).name,
        commit_message="Upload yolov8 tensorboard logs",
        token=hf_token,
        repo_type="model",
    )


def _push_to_hfhub(
    hf_model_id,
    save_dir,
    hf_token=None,
    hf_private=False,
    hf_dataset_id=None,
    input_size=640,
    best_ap50=None,
    task="object-detection",
):
    LOGGER.info(f"Pushing to hf.co/{hf_model_id}")

    push_config_to_hfhub(
        repo_id=hf_model_id,
        exp_folder=save_dir,
        best_ap50=best_ap50,
        input_size=input_size,
        task=task,
        hf_token=hf_token,
        private=hf_private,
    )
    push_model_card_to_hfhub(
        repo_id=hf_model_id,
        exp_folder=save_dir,
        input_size=input_size,
        task=task,
        hf_token=hf_token,
        private=hf_private,
        dataset_id=hf_dataset_id,
        ap50=best_ap50,
    )
    push_model_to_hfhub(
        repo_id=hf_model_id, exp_folder=save_dir, hf_token=hf_token, private=hf_private
    )


def push_to_hfhub(
    exp_dir,
    hf_model_id,
    hf_token=None,
    hf_private=False,
    hf_dataset_id=None,
):
    from ultralytics import YOLO

    best_weight_path = Path(exp_dir) / "weights" / "best.pt"
    model = YOLO(model=best_weight_path)

    # read the largest value in metrics/mAP50(B) column from csv file named results.csv
    df = pd.read_csv(Path(exp_dir) / "results.csv")
    df = df.rename(columns=lambda x: x.strip())
    best_ap50 = df["metrics/mAP50(B)"].max()

    if model.overrides["task"] == "detect":
        task = "object-detection"
    else:
        raise RuntimeError(f"Task {model.overrides['task']} is not supported")

    _push_to_hfhub(
        hf_model_id=hf_model_id,
        hf_token=hf_token,
        hf_private=hf_private,
        save_dir=exp_dir,
        hf_dataset_id=hf_dataset_id,
        input_size=model.overrides["imgsz"],
        best_ap50=best_ap50,
        task=task,
    )


def donwload_from_hub(hf_model_id, hf_token=None):
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_files = list_repo_files(repo_id=hf_model_id, repo_type="model", token=hf_token)

    # download config file for triggering download counter
    config_file = "config.json"
    if config_file in repo_files:
        _ = hf_hub_download(
            repo_id=hf_model_id,
            filename=config_file,
            repo_type="model",
            token=hf_token,
        )

    # download model file
    model_file = [f for f in repo_files if f.endswith(".pt")][0]
    file = hf_hub_download(
        repo_id=hf_model_id,
        filename=model_file,
        repo_type="model",
        token=hf_token,
    )
    return file
