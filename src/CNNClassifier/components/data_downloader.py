import os
import kaggle
from ..logger import logger


class DataDownloader:
    def __init__(self, kaggle_dataset_path: str, data_folder_path: str):
        self.kaggle_dataset_path = kaggle_dataset_path
        self.data_folder_path = data_folder_path
        self._prepare_download()
                
    def _prepare_download(self):
        # Check & Create the data directory
        os.makedirs(self.data_folder_path, exist_ok=True)

        try:
            self.kaggle_api = kaggle.api
            self.kaggle_api.authenticate()
            print("Kaggle API authenticated successfully")
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            print("Kaggle API authentication failed")
            logger.info("Kaggle API authentication failed")
            raise e
        
    def download_data(self):
        try:
            self.kaggle_api.dataset_download_files(
                dataset=self.kaggle_dataset_path, 
                path=self.data_folder_path, 
                unzip=True)
            print(f"Data downloaded successfully to {self.data_folder_path}")
            logger.info(f"Data downloaded successfully to {self.data_folder_path}")
        except Exception as e:
            print("Data download failed")
            logger.info("Data download failed")
            raise e
