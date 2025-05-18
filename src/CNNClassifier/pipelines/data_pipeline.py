from CNNClassifier.components.data_downloader import DataDownloader
from CNNClassifier.components.image_dataset import ImageDataset
from CNNClassifier.logger import logger
from CNNClassifier.config import DataPipelineConfig


class DataPipeline:
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        self.data_pipeline_config: DataPipelineConfig = data_pipeline_config        
        self.data_downloader = DataDownloader(self.data_pipeline_config.data_downloader_config)
        self.train_dataset: ImageDataset = None
        self.val_dataset: ImageDataset = None
        self.test_dataset: ImageDataset = None
        
    def _download_and_split_data(self) -> None:
        self.data_downloader.download_data()
        self.data_downloader.split_data()
        
    def _create_datasets(self) -> None:
        self.train_dataset = ImageDataset(
            data_path=self.data_pipeline_config.train_data_folder_path / "train_data.csv",
            images_path=self.data_pipeline_config.train_data_folder_path / "images",
            dataset_type="train"
        )
        self.val_dataset = ImageDataset(
            data_path=self.data_pipeline_config.val_data_folder_path / "val_data.csv",
            images_path=self.data_pipeline_config.val_data_folder_path / "images",
            dataset_type="val"
        )
        self.test_dataset = ImageDataset(
            data_path=self.data_pipeline_config.test_data_folder_path / "test_data.csv",
            images_path=self.data_pipeline_config.test_data_folder_path / "images",
            dataset_type="test"
        )
    
    def _save_datasets(self) -> None:
        self.train_dataset.export_dataset(self.data_pipeline_config.train_data_folder_path / "train_data.pt")
        self.val_dataset.export_dataset(self.data_pipeline_config.val_data_folder_path / "val_data.pt")
        self.test_dataset.export_dataset(self.data_pipeline_config.test_data_folder_path / "test_data.pt")
    
    def run_pipeline(self) -> None:
        logger.info("Starting data pipeline")
        self._download_and_split_data()
        self._create_datasets()
        self._save_datasets()
        logger.info("Data pipeline completed")