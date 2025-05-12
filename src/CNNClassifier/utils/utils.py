import os
from box.exceptions import BoxValueError
import yaml
from ..logger import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Dict
import base64
import cv2
import matplotlib.pyplot as plt

from ..logger import logger


def read_csv(csv_path: Path, header: bool = True) -> List[Dict[Any, Any]]:
    try:
        with open(csv_path, "r") as file:
            lines = [line.strip().split(",") for line in file.readlines()]
            logger.info(f"CSV file: {csv_path} loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"CSV file not found at {csv_path}")
        raise e
    
    data = []
    if header:
        colnames = lines[0]
        for line in lines[1:]:
            item = {colname: line[i] for i, colname in enumerate(colnames)}
            data.append(item)
    else:
        colnames = ["col_" + str(i) for i, _ in enumerate(lines[0])]
        for line in lines:
            item = {colname: line[i] for i, colname in enumerate(colnames)}
            data.append(item)
    return data

def read_yaml(path_to_yaml: Path) -> Dict[Any, Any]:
    try:
        with open("path_to_yaml", "r") as file:
            loaded_yaml = yaml.safe_load(file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return loaded_yaml
    except FileNotFoundError as e:
        logger.error(f"Yaml file not found at {path_to_yaml}")
        raise FileNotFoundError("yaml file is empty")
    except Exception as e:
        raise e
    
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

def show_image(image_path: str) -> None:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()