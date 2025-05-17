from CNNClassifier.components.model_trainer import ModelTrainer
from CNNClassifier.config import TrainingPipelineConfig
from CNNClassifier.logger import logger


class TrainingPipeline:
    def __init__(self,
                 model_trainer: ModelTrainer):
        self.model_trainer = model_trainer
        self.dataset_factory = DatasetFactory()

    def _download_data(self) -> None:
        self.data_downloader.download_data()
        
    def _prepare_datasets(self) -> None:
        self.train_loader, self.val_loader, self.test_loader = self.dataset_factory.get_datasets()
        
    def _export_datasets(self) -> None:
        self.export_path = self.data_downloader.data_folder_path
        
    
    def run_pipeline(self) -> None:
        self._download_data()
        self._prepare_datasets()
        self._export_datasets()