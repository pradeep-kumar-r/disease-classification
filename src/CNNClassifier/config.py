from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
from CNNClassifier.utils import read_yaml


class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            load_dotenv()
            cls.config = read_yaml(Path("config.yaml"))
        return cls._instance
    
    def get_config(self):
        return self.config
    

config_manager = ConfigManager()


@dataclass(frozen=True)
class DataDownloaderConfig:
    kaggle_dataset_path: Path = Path(config_manager.get_config()['kaggle_dataset_path'])
    data_folder_path: Path = Path(config_manager.get_config()['data_folder_path'])
    kaggle_user: str = os.getenv("KAGGLE_USER")
    kaggle_key: str = os.getenv("KAGGLE_KEY")
    train_data_folder_path: Path = Path(config_manager.get_config()['train_data_folder_path'])
    val_data_folder_path: Path = Path(config_manager.get_config()['val_data_folder_path'])
    test_data_folder_path: Path = Path(config_manager.get_config()['test_data_folder_path'])
    train_split: float = config_manager.get_config()['train_split']
    val_split: float = config_manager.get_config()['val_split']
    

@dataclass(frozen=True)
class DataPipelineConfig:
    data_downloader_config: DataDownloaderConfig = DataDownloaderConfig()
    data_folder_path: Path = Path(config_manager.get_config()['data_folder_path'])
    images_path: Path = Path(config_manager.get_config()['images_folder_path'])
    batch_size: int = config_manager.get_config()['batch_size']
    dataset_path: Path = Path(config_manager.get_config()['dataset_path'])
    train_data_folder_path: Path = Path(config_manager.get_config()['train_data_folder_path'])
    val_data_folder_path: Path = Path(config_manager.get_config()['val_data_folder_path'])
    test_data_folder_path: Path = Path(config_manager.get_config()['test_data_folder_path'])

    
@dataclass(frozen=True)
class ArtefactsConfig:
    artefacts_path: Path = Path(config_manager.get_config()['artefacts_folder_path'])


@dataclass(frozen=True)
class ModelTrainingConfig:
    num_epochs: int = config_manager.get_config()['num_epochs']
    learning_rate: float = config_manager.get_config()['learning_rate']
    batch_size: int = config_manager.get_config()['batch_size']
    
    
@dataclass(frozen=True)
class TrainingPipelineConfig:
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    artefacts_config: ArtefactsConfig = ArtefactsConfig()
    train_dataset_path: Path = Path(config_manager.get_config()['train_data_folder_path']) / "train_data.pt"
    val_dataset_path: Path = Path(config_manager.get_config()['val_data_folder_path']) / "val_data.pt"
    test_dataset_path: Path = Path(config_manager.get_config()['test_data_folder_path']) / "test_data.pt"
    
    

# Testing configs
if __name__ == "__main__":
    print(DataDownloaderConfig().data_folder_path)