import logging
import os
from pathlib import Path
from ultralytics import YOLO as YOLOBase
from ultralytics.nn.tasks import attempt_load_weights

from yolov8tohf.hf_utils import donwload_from_hub

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)
LOGGER = logging.getLogger(__name__)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class YOLO(YOLOBase):
    def __init__(self, model="yolov8n.yaml", type="v8", hf_token=None) -> None:
        """
        Initializes the YOLO object.

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
        self.ckpt_path = None
        self.cfg = None  # if loaded from *.yaml
        self.overrides = {}  # overrides for trainer object
        self.init_disabled = False  # disable model initialization

        # Load or create new YOLO model
        if Path(model).suffix == "":
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
        weights = donwload_from_hub(weights, hf_token=hf_token)

        self.model = attempt_load_weights(weights)
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
