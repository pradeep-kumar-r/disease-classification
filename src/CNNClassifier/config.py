from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
from CNNClassifier.utils import read_yaml


load_dotenv()
config = read_yaml(Path("config.yaml"))


@dataclass(frozen=True)
class DataDownloaderConfig:
    kaggle_dataset_path: Path = Path(config['kaggle_dataset_path'])
    data_folder_path: Path = Path(config['data_folder_path'])
    kaggle_user: str = os.getenv("KAGGLE_USER")
    kaggle_key: str = os.getenv("KAGGLE_KEY")
    train_data_folder_path: Path = Path(config['train_data_folder_path'])
    val_data_folder_path: Path = Path(config['val_data_folder_path'])
    test_data_folder_path: Path = Path(config['test_data_folder_path'])
    train_split: float = config['train_split']
    val_split: float = config['val_split']
    

@dataclass(frozen=True)
class DataPipelineConfig:
    data_downloader_config: DataDownloaderConfig = DataDownloaderConfig()
    data_folder_path: Path = Path(config['data_folder_path'])
    images_path: Path = Path(config['images_folder_path'])
    batch_size: int = config['batch_size']
    dataset_path: Path = Path(config['dataset_path'])
    train_data_folder_path: Path = Path(config['train_data_folder_path'])
    val_data_folder_path: Path = Path(config['val_data_folder_path'])
    test_data_folder_path: Path = Path(config['test_data_folder_path'])

    
@dataclass(frozen=True)
class ArtefactsConfig:
    artefacts_path: Path = Path(config['artefacts_folder_path'])


@dataclass(frozen=True)
class ModelTrainingConfig:
    num_epochs: int = config['num_epochs']
    learning_rate: float = config['learning_rate']
    
    
@dataclass(frozen=True)
class TrainingPipelineConfig:
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    artefacts_config: ArtefactsConfig = ArtefactsConfig()
    
    

# Testing configs
if __name__ == "__main__":
    print(DataDownloaderConfig().data_folder_path)