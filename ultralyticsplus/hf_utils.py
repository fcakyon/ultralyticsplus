import logging
import os
from pathlib import Path

import pandas as pd
from sahi.utils.cv import read_image_as_pil

from ultralyticsplus.other_utils import add_text_to_image

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)
LOGGER = logging.getLogger(__name__)


def generate_model_usage_markdown(
    repo_id,
    score_map50=None,
    score_map50_mask=None,
    score_top1_acc=None,
    score_top5_acc=None,
    model_type="v8",
    task="object-detection",
    dataset_id=None,
    labels=None,
    custom_tags=None,
):
    from ultralytics import __version__ as ultralytics_version

    from ultralyticsplus import __version__ as ultralyticsplus_version

    hf_task = "image-segmentation" if task == "instance-segmentation" else task

    model_str = "yolo" + model_type

    if hf_task == "image-segmentation":
        import_str = "from ultralyticsplus import YOLO, render_result"
        postprocess_str = """print(results[0].boxes)
print(results[0].masks)
render = render_result(model=model, image=image, result=results[0])
render.show()"""
        model_params_str = """model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image"""
        metrics_str = f"""      - type: precision  # since mAP@0.5 is not available on hf.co/metrics
        value: {score_map50}  # min: 0.0 - max: 1.0
        name: mAP@0.5(box)
      - type: precision  # since mAP@0.5 is not available on hf.co/metrics
        value: {score_map50_mask}  # min: 0.0 - max: 1.0
        name: mAP@0.5(mask)"""

    elif hf_task == "object-detection":
        import_str = "from ultralyticsplus import YOLO, render_result"
        postprocess_str = """print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()"""
        model_params_str = """model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image"""
        metrics_str = f"""      - type: precision  # since mAP@0.5 is not available on hf.co/metrics
        value: {score_map50}  # min: 0.0 - max: 1.0
        name: mAP@0.5(box)"""

    elif hf_task == "image-classification":
        import_str = "from ultralyticsplus import YOLO, postprocess_classify_output"
        postprocess_str = """print(results[0].probs) # [0.1, 0.2, 0.3, 0.4]
processed_result = postprocess_classify_output(model, result=results[0])
print(processed_result) # {"cat": 0.4, "dog": 0.6}"""
        model_params_str = (
            """model.overrides['conf'] = 0.25  # model confidence threshold"""
        )
        metrics_str = f"""      - type: accuracy
        value: {score_top1_acc}  # min: 0.0 - max: 1.0
        name: top1 accuracy
      - type: accuracy
        value: {score_top5_acc}  # min: 0.0 - max: 1.0
        name: top5 accuracy"""

    custom_tags_str = ""
    if custom_tags:
        if not isinstance(custom_tags, list):
            custom_tags = [custom_tags]
        for ind, custom_tag in enumerate(custom_tags):
            custom_tags_str += f"- {custom_tag}"
            if ind != len(custom_tags) - 1:
                custom_tags_str += "\n"

    if dataset_id is not None:
        datasets_str_1 = f"""
datasets:
- {dataset_id}
"""
        datasets_str_2 = f"""
    dataset:
      type: {dataset_id}
      name: {dataset_id.split("/")[-1]}
      split: validation
"""
    else:
        datasets_str_1 = datasets_str_2 = ""
    return f""" 
---
tags:
- ultralyticsplus
- {model_str}
- ultralytics
- yolo
- vision
- {hf_task}
- pytorch
{custom_tags_str}
library_name: ultralytics
library_version: {ultralytics_version}
inference: false
{datasets_str_1}
model-index:
- name: {repo_id}
  results:
  - task:
      type: {hf_task}
{datasets_str_2}
    metrics:
{metrics_str}
---

<div align="center">
  <img width="640" alt="{repo_id}" src="https://huggingface.co/{repo_id}/resolve/main/thumbnail.jpg">
</div>

### Supported Labels

```
{labels}
```

### How to use

- Install [ultralyticsplus](https://github.com/fcakyon/ultralyticsplus):

```bash
pip install ultralyticsplus=={ultralyticsplus_version} ultralytics=={ultralytics_version}
```

- Load model and perform prediction:

```python
{import_str}

# load model
model = YOLO('{repo_id}')

# set model parameters
{model_params_str}

# set image
image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model.predict(image)

# observe results
{postprocess_str}
```

"""


def generate_thumbnail(
    image_path_or_url,
    repo_id=None,
    task="object-detection",
    thumbnail_text=None,
    export_dir=None,
):
    """
    Generate thumbnail for the model card

    USERNAME/yolov8n-garbage > YOLOv8 Garbage Detection
    """
    if str(image_path_or_url).startswith("http") and not export_dir:
        raise ValueError("export_dir must be specified for remote images.")

    if thumbnail_text:
        pass
    elif repo_id:
        thumbnail_text = repo_id.split("/")[-1]
        texts = thumbnail_text.split("-")
        for ind, text in enumerate(texts):
            if "yolo" not in text.lower():
                text = text.title()
            texts[ind] = text.replace("yolo", "YOLO")

        thumbnail_text = " ".join(texts)

        if task == "object-detection":
            thumbnail_text += " Detection"
        elif task == "image-classification":
            thumbnail_text += " Classification"
        elif task == "instance-segmentation":
            thumbnail_text += " Segmentation"
        else:
            raise ValueError(f"Task {task} is not supported.")
    else:
        raise ValueError("repo_id or thumbnail_text must be provided.")

    image = add_text_to_image(
        text=thumbnail_text,
        pil_image=read_image_as_pil(image_path_or_url),
        brightness=0.60,
        text_font_size=65,
        crop_margin=None,
    )

    if str(image_path_or_url).startswith("http"):
        thumbnail_path = Path(export_dir) / "thumbnail.jpg"
    else:
        thumbnail_path = Path(image_path_or_url).parent / "thumbnail.jpg"
    image.save(str(thumbnail_path), quality=100)

    return thumbnail_path


def push_model_card_to_hfhub(
    repo_id,
    exp_folder,
    hf_token=None,
    task="object-detection",
    private=False,
    dataset_id=None,
    score_map50=None,
    score_map50_mask=None,
    score_top1_acc=None,
    score_top5_acc=None,
    model_type="v8",
    thumbnail_text=None,
    custom_tags=None,
):
    from huggingface_hub import create_repo, upload_file
    from ultralytics import YOLO

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )

    # upload thumbnail to the repo
    if task in ["object-detection", "instance-segmentation"]:
        sample_visual_path = str(Path(exp_folder) / "val_batch0_labels.jpg")
    elif task == "image-classification":
        sample_visual_path = "https://user-images.githubusercontent.com/34196005/212529509-3723ef83-e184-4e57-af37-ed7cfe0faf11.jpg"
    else:
        raise ValueError(f"Task {task} is not supported.")

    thumbnail_path = generate_thumbnail(
        sample_visual_path,
        repo_id=repo_id,
        task=task,
        thumbnail_text=thumbnail_text,
        export_dir=exp_folder,
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(thumbnail_path),
        path_in_repo="thumbnail.jpg",
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
        dataset_id=dataset_id,
        score_map50=score_map50,
        score_map50_mask=score_map50_mask,
        score_top1_acc=score_top1_acc,
        score_top5_acc=score_top5_acc,
        model_type=model_type,
        labels=labels,
        custom_tags=custom_tags,
    )
    modelcard_path = Path(exp_folder) / "README.md"
    with open(modelcard_path, "w") as file_object:
        file_object.write(modelcard_markdown)
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(modelcard_path),
        path_in_repo=Path(modelcard_path).name,
        commit_message="add ultralytics model card",
        token=hf_token,
        repo_type="model",
    )


def push_config_to_hfhub(
    repo_id,
    exp_folder,
    score_map50=None,
    score_map50_mask=None,
    score_top1_acc=None,
    score_top5_acc=None,
    input_size=640,
    task="object-detection",
    hf_token=None,
    private=False,
    model_type="v8",
):
    """
    Pushes a yolov5 config to huggingface hub

    Arguments:
        repo_id (str): The name of the repository to create on huggingface.co
        exp_folder (str): The path to the experiment folder
        input_size (int): The input size of the model (default: 640)
        task (str): The task of the model (default: object-detection)
        hf_token (str): The huggingface token to use to push the model
        private (bool): Whether the model should be private or not
        model_type (str): The type of the model (default: v8)
    """
    import json

    from huggingface_hub import create_repo, upload_file
    from ultralytics import __version__ as ultralytics_version

    from ultralyticsplus import __version__ as ultralyticsplus_version

    # create config
    config = {
        "input_size": input_size,
        "task": task,
        "ultralyticsplus_version": ultralyticsplus_version,
        "ultralytics_version": ultralytics_version,
        "model_type": model_type,
    }
    if score_map50 is not None:
        config["score_map50"] = score_map50
    if score_map50_mask is not None:
        config["score_map50_mask"] = score_map50_mask
    if score_top1_acc is not None:
        config["score_top1_acc"] = score_top1_acc
    if score_top5_acc is not None:
        config["score_top5_acc"] = score_top5_acc

    # save config
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
        commit_message="add ultralyticsplus config",
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
    from glob import glob

    from huggingface_hub import create_repo, delete_file, list_repo_files, upload_file

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
    score_map50=None,
    score_map50_mask=None,
    score_top1_acc=None,
    score_top5_acc=None,
    task="object-detection",
    model_type="v8",
    thumbnail_text=None,
    custom_tags=None,
):
    LOGGER.info(f"Pushing to hf.co/{hf_model_id}")

    push_config_to_hfhub(
        repo_id=hf_model_id,
        exp_folder=save_dir,
        score_map50=score_map50,
        score_map50_mask=score_map50_mask,
        score_top1_acc=score_top1_acc,
        score_top5_acc=score_top5_acc,
        input_size=input_size,
        task=task,
        hf_token=hf_token,
        private=hf_private,
        model_type=model_type,
    )
    push_model_card_to_hfhub(
        repo_id=hf_model_id,
        exp_folder=save_dir,
        task=task,
        hf_token=hf_token,
        private=hf_private,
        dataset_id=hf_dataset_id,
        score_map50=score_map50,
        score_map50_mask=score_map50_mask,
        score_top1_acc=score_top1_acc,
        score_top5_acc=score_top5_acc,
        model_type=model_type,
        thumbnail_text=thumbnail_text,
        custom_tags=custom_tags,
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
    thumbnail_text=None,
    custom_tags=None,
    return_dict=False,
):
    """
    Pushes a ultralytics model to huggingface hub

    Args:
        exp_dir (str): ultralytics experiment folder path
        hf_model_id (str): huggingface model id to be uploaded to
        hf_token (str): huggingface write token
        hf_private (bool): whether to make the repo private or not
        hf_dataset_id (str): huggingface dataset id to be used for model card
        thumbnail_text (str): text to be used in thumbnail
        custom_tags (list): list of custom tags to be used for model card
        return_dict (bool): whether to return a dictionary of results or not
    """
    from ultralytics import YOLO

    best_weight_path = Path(exp_dir) / "weights" / "best.pt"
    model = YOLO(model=best_weight_path)

    results_df = pd.read_csv(Path(exp_dir) / "results.csv")
    results_df = results_df.rename(columns=lambda x: x.strip())

    if model.overrides["task"] == "detect":
        task = "object-detection"
    elif model.overrides["task"] == "segment":
        task = "instance-segmentation"
    elif model.overrides["task"] == "classify":
        task = "image-classification"
    else:
        raise RuntimeError(f"Task {model.overrides['task']} is not supported")

    score_map50 = None
    score_map50_mask = None
    score_top1_acc = None
    score_top5_acc = None
    if task in ["object-detection", "instance-segmentation"]:
        # read the largest value in metrics/mAP50(B) column from csv file named results.csv
        score_map50 = results_df["metrics/mAP50(B)"].max().item()
    if task == "instance-segmentation":
        # read the largest value in metrics/mAP50(B) metrics/mAP50(M) columns from csv file named results.csv
        score_map50_mask = results_df["metrics/mAP50(M)"].max().item()
    if task == "image-classification":
        # read the largest value in metrics/accuracy_top1 metrics/accuracy_top5 columns from csv file named results.csv
        score_top1_acc = results_df["metrics/accuracy_top1"].max().item()
        score_top5_acc = results_df["metrics/accuracy_top5"].max().item()

    _push_to_hfhub(
        hf_model_id=hf_model_id,
        hf_token=hf_token,
        hf_private=hf_private,
        save_dir=exp_dir,
        hf_dataset_id=hf_dataset_id,
        input_size=model.overrides["imgsz"],
        score_map50=score_map50,
        score_map50_mask=score_map50_mask,
        score_top1_acc=score_top1_acc,
        score_top5_acc=score_top5_acc,
        task=task,
        thumbnail_text=thumbnail_text,
        custom_tags=custom_tags,
    )

    if return_dict:
        return {
            "score_map50": score_map50,
            "score_map50_mask": score_map50_mask,
            "score_top1_acc": score_top1_acc,
            "score_top5_acc": score_top5_acc,
            "task": task,
            "model_type": model.type,
            "thumbnail_url": f"https://huggingface.co/{hf_model_id}/resolve/main/thumbnail.jpg",
        }


def download_from_hub(hf_model_id, hf_token=None):
    """
    Downloads a model from huggingface hub

    Args:
        hf_model_id (str): huggingface model id to be downloaded from
        hf_token (str): huggingface read token

    Returns:
        model_path (str): path to downloaded model
    """
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
