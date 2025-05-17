import os
import kaggle
import pandas as pd
from pathlib import Path
import shutil
from CNNClassifier.logger import logger
from CNNClassifier.config import DataDownloaderConfig
from CNNClassifier.utils import rename_file, rename_folder, move_copy_file, create_directories


class DataDownloader:
    _instance = None
    
    def __new__(cls, data_downloader_config: DataDownloaderConfig | None = None):
        if cls._instance is None:
            if data_downloader_config is None:
                raise ValueError("DataDownloaderConfig is required for first initialization")
            cls._instance = super(DataDownloader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_downloader_config: DataDownloaderConfig | None= None):
        if self._initialized:
            logger.error("Singleton class already initialized")
            return
            
        if data_downloader_config is None:
            logger.error("DataDownloaderConfig is required for first initialization")
            raise ValueError("DataDownloaderConfig is required for first initialization")
            
        ### Can be commented out for actual run
        if data_downloader_config.data_folder_path.exists():
            shutil.rmtree(data_downloader_config.data_folder_path)
            os.makedirs(data_downloader_config.data_folder_path, exist_ok=True)
        
        self.data_downloader_config = data_downloader_config
        os.makedirs(self.data_downloader_config.data_folder_path, exist_ok=True)
        self._authenticate()
        print(f"Data folder path: {self.data_downloader_config.data_folder_path}")
        print(f"Kaggle dataset path: {self.data_downloader_config.kaggle_dataset_path}")
        self._initialized = True
    
    def _authenticate(self) -> None:
        try:
            kaggle.api.authenticate()
            print("Kaggle API authenticated successfully")
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            print("Kaggle API authentication failed")
            logger.info("Kaggle API authentication failed")
            raise e
        
    def download_data(self) -> None:
        try:
            kaggle.api.dataset_download_files(
                dataset=str(self.data_downloader_config.kaggle_dataset_path), 
                path=str(self.data_downloader_config.data_folder_path), 
                unzip=True)
            rename_file(Path(self.data_downloader_config.data_folder_path) / "train_data.csv", "data")
            rename_folder(Path(self.data_downloader_config.data_folder_path) / "Train", "images")
            print(f"Data downloaded successfully to {self.data_downloader_config.data_folder_path}")
            logger.info(f"Data downloaded successfully to {self.data_downloader_config.data_folder_path}")
        except Exception as e:
            print("Data download failed")
            logger.info("Data download failed")
            raise e
        
    def split_data(self) -> None:
        try:
            df = pd.read_csv(str(Path(self.data_downloader_config.data_folder_path) / "data.csv"))
            total_size = len(df)
            train_size = int(total_size * DataDownloaderConfig.train_split)
            val_size = int(total_size * DataDownloaderConfig.val_split)
            
            df = df.sample(frac=1, random_state=42) # Random shuffling
            train_df = df[:train_size]
            val_df = df[train_size:train_size + val_size]
            test_df = df[train_size + val_size:]
            
            create_directories([self.data_downloader_config.train_data_folder_path,
                                self.data_downloader_config.val_data_folder_path,
                                self.data_downloader_config.test_data_folder_path])
            
            train_df.to_csv(str(self.data_downloader_config.train_data_folder_path / "train_data.csv"), index=False)
            self._copy_files(train_df, self.data_downloader_config.train_data_folder_path / "images")
            
            val_df.to_csv(str(self.data_downloader_config.val_data_folder_path / "val_data.csv"), index=False)
            self._copy_files(val_df, self.data_downloader_config.val_data_folder_path / "images")
            
            test_df.to_csv(str(self.data_downloader_config.test_data_folder_path / "test_data.csv"), index=False)
            self._copy_files(test_df, self.data_downloader_config.test_data_folder_path / "images")
            
            logger.info("Train, Val & Test data folders & files split successfully")
            
        except Exception as e:
            logger.error(f"Error in splitting Train, Val & Test data: {e}")
            raise e
            
    def _copy_files(self, df: pd.DataFrame, target_dir: Path) -> None:
        for _, row in df.iterrows():
            image_name = row['images']
            src_path = Path(self.data_downloader_config.data_folder_path) / "images" / image_name
            dst_dir = target_dir
            
            if src_path.exists():
                move_copy_file(src_path, dst_dir)
            else:
                logger.warning(f"Image not found: {src_path}")