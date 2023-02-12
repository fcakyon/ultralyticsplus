import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sahi.prediction import ObjectPrediction, PredictionScore
from sahi.utils.cv import (get_bool_mask_from_coco_segmentation,
                           read_image_as_pil, visualize_object_predictions)
from ultralytics import YOLO as YOLOBase
from ultralytics.nn.tasks import attempt_load_one_weight

from ultralyticsplus.hf_utils import download_from_hub

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)
LOGGER = logging.getLogger(__name__)


class YOLO(YOLOBase):
    def __init__(self, model="yolov8n.yaml", type="v8", hf_token=None) -> None:
        """
        Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
            hf_token (str): huggingface token
        """
        self.type = type
        self.ModelClass = None  # model class
        self.TrainerClass = None  # trainer class
        self.ValidatorClass = None  # validator class
        self.PredictorClass = None  # predictor class
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object

        # Load or create new YOLO model
        if Path(model).suffix not in (".pt", ".yaml"):
            self._load_from_hf_hub(model, hf_token=hf_token)
        else:
            {".pt": self._load, ".yaml": self._new}[Path(model).suffix](model)

    def _load_from_hf_hub(self, weights: str, hf_token=None):
        """
        Initializes a new model and infers the task type from the model head

        Args:
            weights (str): model checkpoint to be loaded
            hf_token (str): huggingface token
        """
        # try to download from hf hub
        weights = download_from_hub(weights, hf_token=hf_token)

        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.ckpt_path = weights
        self.task = self.model.args["task"]
        self.overrides = self.model.args
        self._reset_ckpt_args(self.overrides)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)


def render_result(
    image, model: YOLO, result: "ultralytics.yolo.engine.result.Result"
) -> Image.Image:
    """
    Renders predictions on the image

    Args:
        image (str, URL, Image.Image): image to be rendered
        model (YOLO): YOLO model
        result (ultralytics.yolo.engine.result.Result): output of the model. This is the output of the model.predict() method.

    Returns:
        Image.Image: Image with predictions
    """
    if model.overrides["task"] not in ["detect", "segment"]:
        raise ValueError(
            f"Model task must be either 'detect' or 'segment'. Got {model.overrides['task']}"
        )

    image = read_image_as_pil(image)
    np_image = np.ascontiguousarray(image)

    names = model.model.names

    masks = result.masks
    boxes = result.boxes

    object_predictions = []
    if boxes is not None:
        det_ind = 0
        for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if masks:
                img_height = np_image.shape[0]
                img_width = np_image.shape[1]
                segments = masks.segments
                segments = segments[det_ind]  # segments: np.array([[x1, y1], [x2, y2]])
                # convert segments into full shape
                segments[:, 0] = segments[:, 0] * img_width
                segments[:, 1] = segments[:, 1] * img_height
                segmentation = [segments.ravel().tolist()]

                bool_mask = get_bool_mask_from_coco_segmentation(
                    segmentation, width=img_width, height=img_height
                )
                if sum(sum(bool_mask == 1)) <= 2:
                    continue
                object_prediction = ObjectPrediction.from_coco_segmentation(
                    segmentation=segmentation,
                    category_name=names[int(cls)],
                    category_id=int(cls),
                    full_shape=[img_height, img_width],
                )
                object_prediction.score = PredictionScore(value=conf)
            else:
                object_prediction = ObjectPrediction(
                    bbox=xyxy.tolist(),
                    category_name=names[int(cls)],
                    category_id=int(cls),
                    score=conf,
                )
            object_predictions.append(object_prediction)
            det_ind += 1

    result = visualize_object_predictions(
        image=np_image,
        object_prediction_list=object_predictions,
    )

    return Image.fromarray(result["image"])


def postprocess_classify_output(
    model: YOLO, result: "ultralytics.yolo.engine.result.Result"
) -> dict:
    """
    Postprocesses the output of classification models

    Args:
        model (YOLO): YOLO model
        prob (np.ndarray): output of the model

    Returns:
        dict: dictionary of outputs with labels
    """
    output = {}
    if isinstance(model.model.names, list):
        names = model.model.names
    elif isinstance(model.model.names, dict):
        names = model.model.names.values()
    else:
        raise ValueError("Model names must be either a list or a dict")

    for i, label in enumerate(names):
        output[label] = result.probs[i].item()
    return output
