from CNNClassifier.logger import logger
from CNNClassifier.components.model import BasicCNNModel
from CNNClassifier.pipelines.data_pipeline import DataPipeline
from CNNClassifier.pipelines.training_pipeline import TrainingPipeline
from CNNClassifier.config import DataPipelineConfig, TrainingPipelineConfig


def main() -> None:
    logger.info("\n\n*****\nNEW RUN\n*****")
    
    logger.info("\n\nStep 1: Data Pipeline")
    data_pipeline = DataPipeline(DataPipelineConfig)
    data_pipeline.run_pipeline()
    num_classes = data_pipeline.train_dataset.num_classes()

    logger.info("\n\nStep 2: Training Pipeline")
    training_pipeline = TrainingPipeline(model=BasicCNNModel(num_classes=num_classes),
                                         training_pipeline_config=TrainingPipelineConfig()
                        )
    training_pipeline.run_pipeline()


if __name__ == "__main__":
    main()