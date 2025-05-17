from CNNClassifier.components.data_downloader import DataDownloader
from CNNClassifier.components.dataset_loader import DatasetLoader
from CNNClassifier.logger import logger
from CNNClassifier.config import DataPipelineConfig


class DataPipeline:
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        self.data_pipeline_config = data_pipeline_config
        
        
    def _download_and_split_data(self) -> None:
        self.data_downloader = DataDownloader(self.data_pipeline_config.data_downloader_config)
        self.data_downloader.download_data()
        self.data_downloader.split_data()
        
    def _prepare_datasets(self) -> None:
        self.train_data_loader = DatasetLoader( 
                 dataset_path= self.data_pipeline_config.train_data_folder_path / "train_data.csv",
                 batch_size= 32,
                 shuffle= True,
                 name= "train")
        self.val_data_loader = DatasetLoader( 
                 dataset_path= self.data_pipeline_config.val_data_folder_path / "val_data.csv",
                 name= "val")
        self.test_data_loader = DatasetLoader( 
                 dataset_path= self.data_pipeline_config.test_data_folder_path / "test_data.csv",
                 name= "test")
        
    def run_pipeline(self) -> None:
        logger.info("Starting data pipeline")
        self._download_and_split_data()
        self._prepare_datasets()
        logger.info("Data pipeline completed")