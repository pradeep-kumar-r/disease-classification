import os
import yaml
import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Literal
import base64
import cv2
import matplotlib.pyplot as plt
import shutil
from CNNClassifier.logger import logger


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

def read_yaml(filepath: Path) -> Dict[Any, Any]:
    try:
        with open(filepath, "r") as file:
            loaded_yaml = yaml.safe_load(file)
            logger.info(f"yaml file: {filepath} loaded successfully")
            return loaded_yaml
    except FileNotFoundError:
        logger.error(f"No file found at {filepath}")
        raise FileNotFoundError("yaml file is empty")
    except Exception as e:
        logger.error(f"Error loading yaml file: {e}")
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

def rename_file(old_path: Path, new_name: str) -> None:
    try:
        if not old_path.exists():
            logger.error(f"File not found at {old_path}")
            raise FileNotFoundError(f"File not found at {old_path}")
        directory = old_path.parent
        extension = old_path.suffix
        new_path = directory / f"{new_name}{extension}"
        old_path.rename(new_path)
        logger.info(f"Renamed file from {old_path} to {new_path}")
    except Exception as e:
        logger.error(f"Error renaming file: {e}")
        raise e

def rename_folder(old_path: Path, new_name: str) -> None:
    try:
        if not old_path.exists():
            logger.error(f"Folder not found at {old_path}")
            raise FileNotFoundError(f"Folder not found at {old_path}")
        new_path = old_path.parent / new_name
        old_path.rename(new_path)
        logger.info(f"Renamed folder from {old_path} to {new_path}")
    except Exception as e:
        logger.error(f"Error renaming folder: {e}")
        raise e

def move_copy_file(old_path: Path, new_directory: Path, mode: Literal["move", "copy"] = "copy") -> None:
    try:
        if not old_path.exists():
            logger.error(f"File not found at {old_path}")
            raise FileNotFoundError(f"File not found at {old_path}")
        new_directory.mkdir(parents=True, exist_ok=True)
        extension = old_path.suffix
        old_name = old_path.name
        new_path = new_directory / f"{old_name}{extension}"
        if mode == "move":
            shutil.move(str(old_path), str(new_path))
            logger.info(f"Moved file from {old_path} to {new_path}")
        else:
            shutil.copy(str(old_path), str(new_path))
            logger.info(f"Copied file from {old_path} to {new_path}")
    except Exception as e:
        logger.error(f"Error moving and renaming file: {e}")
        raise e