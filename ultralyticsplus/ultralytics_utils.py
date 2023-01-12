import logging
import os
from pathlib import Path
from typing import Union
from ultralytics import YOLO as YOLOBase
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.utils.plotting import colors, Annotator

import numpy as np
from PIL import Image
from sahi.utils.cv import read_image_as_pil

from ultralyticsplus.hf_utils import download_from_hub

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)
LOGGER = logging.getLogger(__name__)


class YOLO(YOLOBase):
    def __init__(self, model="yolov8n.yaml", type="v8", hf_token=None) -> None:
        """
        > Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
        """
        self.type = type
        self.ModelClass = None  # model class
        self.TrainerClass = None  # trainer class
        self.ValidatorClass = None  # validator class
        self.PredictorClass = None  # predictor class
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
        (
            self.ModelClass,
            self.TrainerClass,
            self.ValidatorClass,
            self.PredictorClass,
        ) = self._guess_ops_from_task(self.task)


def render_predictions(
    model: YOLO, img: Union[np.ndarray, str], det: list
) -> Image.Image:
    """
    Renders predictions on the image

    Args:
        model (YOLO): YOLO model
        img (Union[np.ndarray, str]): original image. RGB image or path to image.
        det (dict): predictions. Should be in the format of
            [x1, y1, x2, y2, conf, class]

    Returns:
        Image.Image: Image with predictions
    """
    img = read_image_as_pil(img)
    img = np.ascontiguousarray(img)
    annotator = Annotator(
        im=img,
        line_width=model.overrides["line_thickness"],
        example=str(model.model.names),
    )

    # render results
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        label = (
            None
            if model.overrides["hide_labels"]
            else (
                model.names[c]
                if model.overrides["hide_conf"]
                else f"{model.model.names[c]} {conf:.2f}"
            )
        )
        annotator.box_label(xyxy, label, color=colors(c, True))

    image = Image.fromarray(annotator.result())
    return image


if __name__ == "__main__":
    hf_model_id = "fcakyon/yolov8s-test"
    hf_token = "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"

    model = YOLO(hf_model_id, hf_token=hf_token)

    img = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    visuals = []  # store visualizations
    for result in model.predict(img, return_outputs=True, conf=0.5):
        print(result)
        visual = render_predictions(model, img=img, det=result["det"])
        visuals.append(visual)

    visuals[0].show()
