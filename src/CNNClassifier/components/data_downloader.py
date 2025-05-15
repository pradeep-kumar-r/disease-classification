import os
import kaggle
from CNNClassifier.logger import logger
from CNNClassifier.config import DataDownloaderConfig


# Singleton
class DataDownloader:
    def __init__(self, 
                 kaggle_dataset_path: str=DataDownloaderConfig.kaggle_dataset_path, 
                 data_folder_path: str=DataDownloaderConfig.data_folder_path):
        self.kaggle_dataset_path = kaggle_dataset_path
        self.data_folder_path = data_folder_path
        os.makedirs(self.data_folder_path, exist_ok=True)
        self._authenticate()
        print(f"Data folder path: {self.data_folder_path}")
        print(f"Kaggle dataset path: {self.kaggle_dataset_path}")
    
    def _authenticate(self):
        try:
            kaggle.api.authenticate()
            # kaggle.api.authenticate(username=DataDownloaderConfig.kaggle_user, 
            #                              key=DataDownloaderConfig.kaggle_key)
            print("Kaggle API authenticated successfully")
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            print("Kaggle API authentication failed")
            logger.info("Kaggle API authentication failed")
            raise e
        
    def download_data(self):
        try:
            kaggle.api.dataset_download_files(
                dataset=self.kaggle_dataset_path, 
                path=self.data_folder_path, 
                unzip=True)
            print(f"Data downloaded successfully to {self.data_folder_path}")
            logger.info(f"Data downloaded successfully to {self.data_folder_path}")
        except Exception as e:
            print("Data download failed")
            logger.info("Data download failed")
            raise e
