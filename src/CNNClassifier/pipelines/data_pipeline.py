from CNNClassifier.components.data_downloader import DataDownloader
from CNNClassifier.components.dataset_factory import DatasetFactory
from CNNClassifier.logger import logger


class DataPipeline:
    def __init__(self):
        self.data_downloader = DataDownloader()
        self.dataset_factory = DatasetFactory()

    def _download_data(self) -> None:
        self.data_downloader.download_data()
        
    def _prepare_datasets(self) -> None:
        self.train_loader, self.val_loader, self.test_loader = self.dataset_factory.get_datasets()
        
    def _export_datasets(self) -> None:
        self.export_path = self.data_downloader.data_folder_path
        self.dataset_factory.save_datasets(self.export_path)
    
    def run_pipeline(self) -> None:
        self._download_data()
        self._prepare_datasets()
        self._export_datasets()