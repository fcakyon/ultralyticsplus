import re

from roboflow import Roboflow

from ultralyticsplus.ultralytics_utils import LOGGER


def extract_roboflow_metadata(url: str) -> tuple:
    match = re.search(r'https://(?:app|universe)\.roboflow\.com/([^/]+)/([^/]+)(?:/dataset)?/([^/]+)', url)
    if match:
        workspace_name = match.group(1)
        project_name = match.group(2)
        project_version = match.group(3)
        return workspace_name, project_name, project_version
    else:
        raise ValueError(f"Invalid Roboflow dataset url ❌ "
                         f"Expected: https://universe.roboflow.com/workspace_name/project_name/project_version. "
                         f"Given: {url}")


def push_to_roboflow_universe(
    exp_dir: str,
    roboflow_url: str,
    roboflow_token: str
):
    workspace_name, project_name, project_version = extract_roboflow_metadata(url=roboflow_url)
    rf = Roboflow(api_key=roboflow_token)
    project_version = rf.workspace(workspace_name).project(project_name).version(int(project_version))
    LOGGER.info(f"Uploading model from local: {exp_dir} to Roboflow: {project_version.id}")
    project_version.deploy(model_type="yolov8", model_path=exp_dir)
