from dataclasses import dataclass
from pathlib import Path
from CNNClassifier.utils.utils import read_yaml


config = read_yaml(Path("config.yaml"))


@dataclass(frozen=True)
class DataDownloaderConfig:
    kaggle_dataset_path: str = config['kaggle_dataset_path']
    data_folder_path: Path = config['data_folder_path']
    

@dataclass(frozen=True)
class DataLoaderConfig:
    data_path: Path = config['data_folder_path']
    images_path: Path = config['images_folder_path']
    train_split: float = config['train_split']
    val_split: float = config['val_split']
    batch_size: int = config['batch_size']
    
    
@dataclass(frozen=True)
class ArtefactsConfig:
    artefacts_path: Path = config['artefacts_folder_path']


@dataclass(frozen=True)
class ModelTrainingConfig:
    num_epochs: int = config['num_epochs']
    learning_rate: float = config['learning_rate']
    
    


if __name__ == "__main__":
    print(DataDownloaderConfig().data_folder_path)