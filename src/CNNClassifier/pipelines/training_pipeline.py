import torch.nn as nn
from CNNClassifier.components.model_trainer import ModelTrainer
from CNNClassifier.components.dataset_loader import DatasetLoader
from CNNClassifier.components.image_dataset import ImageDataset
from CNNClassifier.config import TrainingPipelineConfig
from CNNClassifier.logger import logger
from CNNClassifier.components.model import BasicCNNModel
from CNNClassifier.components.model_evaluator import ModelEvaluator


class TrainingPipeline:
    def __init__(self,
                 training_pipeline_config: TrainingPipelineConfig):
        
        self.training_pipeline_config: TrainingPipelineConfig = training_pipeline_config
        self.train_dataloader: DatasetLoader = None
        self.val_dataloader: DatasetLoader = None
        self.test_dataloader: DatasetLoader = None
        self.model_trainer: ModelTrainer = None
        self.model_evaluator: ModelEvaluator = None
        self.model: nn.Module = None
        self.num_classes: int = None
    
    def _set_model(self) -> None:
        self.model = BasicCNNModel(num_classes=self.num_classes)
    
    def _create_dataloaders(self) -> None:
        train_dataset = ImageDataset.load_dataset(self.training_pipeline_config.train_dataset_path)
        self.num_classes = train_dataset.num_classes()
        val_dataset = ImageDataset.load_dataset(self.training_pipeline_config.val_dataset_path)
        test_dataset = ImageDataset.load_dataset(self.training_pipeline_config.test_dataset_path)
        self.train_dataloader = DatasetLoader(dataset=train_dataset,
                                              batch_size=self.training_pipeline_config.model_training_config.batch_size)
        self.val_dataloader = DatasetLoader(dataset=val_dataset)
        self.test_dataloader = DatasetLoader(dataset=test_dataset)
    
    def _train(self) -> None:
        logger.info("Starting Training Pipeline")
        try:
            self.model_trainer.train()
            logger.info(f"{self.model_trainer}")
            logger.info("Training Pipeline completed successfully")
        except Exception as e:
            logger.error(f"Training Pipeline failed: {str(e)}")
            raise e
        
    def _evaluate(self) -> None:
        logger.info("Starting Evaluation")
        try:
            self.model_evaluator.evaluate()
            self.model_evaluator.save_report()
            logger.info("Evaluation completed successfully")
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise e
    
    def run_pipeline(self) -> None:
        self._create_dataloaders()
        self._set_model()
        self.model_trainer = ModelTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            num_epochs=self.training_pipeline_config.model_training_config.num_epochs,
            learning_rate=self.training_pipeline_config.model_training_config.learning_rate,
            model_save_path=self.training_pipeline_config.artefacts_config.artefacts_path
        )
        self._train()
        self.model_evaluator = ModelEvaluator(
            model=self.model,
            test_dataloader=self.test_dataloader,
            report_save_path=self.training_pipeline_config.artefacts_config.artefacts_path / "evaluation_report.txt"
        )
        self._evaluate()