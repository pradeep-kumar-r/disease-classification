import os
import yaml
import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Literal
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
    except FileNotFoundError:
        logger.error(f"Yaml file not found at {path_to_yaml}")
        raise FileNotFoundError("yaml file is empty")
    except Exception as e:
        raise e
    
def create_directories(path_to_directories: List[Path]) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory at: {path}")

def save_json(filepath: Path, data: dict) -> None:
    if not filepath.exists():
        logger.error(f"JSON file not found at {filepath}")
        raise FileNotFoundError(f"JSON file not found at {filepath}")
    
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"JSON file saved at: {filepath}")

def load_json(filepath: Path) -> Dict[Any, Any]:
    if not filepath.exists() or not filepath.endswith(".json"):
        logger.error(f"JSON file not found at {filepath}")
        raise FileNotFoundError(f"JSON file not found at {filepath}")
    
    with open(filepath, "r") as file:
        content = json.load(file)

    logger.info(f"JSON file loaded succesfully from: {filepath}")
    return content

def get_size(filepath: Path, unit: Literal["KB", "MB", "GB"] = "KB") -> float:
    match unit:
        case "KB":
            return round(os.path.getsize(filepath)/1024)
        case "MB":
            return round(os.path.getsize(filepath)/(1024**2))
        case "GB":
            return round(os.path.getsize(filepath)/(1024**3))
        case _:
            logger.error(f"Invalid unit: {unit}")
            raise ValueError(f"Invalid unit: {unit}")

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as file:
        file.write(imgdata)

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as file:
        return base64.b64encode(file.read())

def show_image(image_path: str) -> None:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()