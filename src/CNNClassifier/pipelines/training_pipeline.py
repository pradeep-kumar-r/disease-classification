import torch.nn as nn
from CNNClassifier.components.model_trainer import ModelTrainer
from CNNClassifier.components.dataset_loader import DatasetLoader
from CNNClassifier.components.image_dataset import ImageDataset
from CNNClassifier.config import TrainingPipelineConfig
from CNNClassifier.logger import logger


class TrainingPipeline:
    def __init__(self,
                 model: nn.Module,
                 training_pipeline_config: TrainingPipelineConfig):
        
        self.training_pipeline_config: TrainingPipelineConfig = training_pipeline_config
        self.model: nn.Module = model
        self.train_dataloader: DatasetLoader = None
        self.val_dataloader: DatasetLoader = None
        self.test_dataloader: DatasetLoader = None
        self.model_trainer: ModelTrainer = None
    
    def _create_dataloaders(self) -> None:
        self.train_dataloader = DatasetLoader(ImageDataset.load_dataset(self.training_pipeline_config.train_dataset_path), 
                                              batch_size=self.training_pipeline_config.model_training_config.batch_size)
        self.val_dataloader = DatasetLoader(ImageDataset.load_dataset(self.training_pipeline_config.val_dataset_path))
        self.test_dataloader = DatasetLoader(ImageDataset.load_dataset(self.training_pipeline_config.test_dataset_path))
        
    
    def _train(self) -> None:
        logger.info("Starting Training Pipeline")
        
        try:
            self.model_trainer.train()
            logger.info(f"{self.model_trainer}")
            logger.info("Training Pipeline completed successfully")
        except Exception as e:
            logger.error(f"Training Pipeline failed: {str(e)}")
            raise e
    
    def run_pipeline(self) -> None:
        self._create_dataloaders()
        self.model_trainer = ModelTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            num_epochs=self.training_pipeline_config.model_training_config.num_epochs,
            learning_rate=self.training_pipeline_config.model_training_config.learning_rate,
            model_save_path=self.training_pipeline_config.artefacts_config.artefacts_path
        )
        self._train()