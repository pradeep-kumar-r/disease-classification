from CNNClassifier.logger import logger
from CNNClassifier.pipelines.data_pipeline import DataPipeline
from CNNClassifier.pipelines.training_pipeline import TrainingPipeline
from CNNClassifier.config import DataPipelineConfig, TrainingPipelineConfig


def main() -> None:
    logger.info("\n\n*****\nNEW RUN\n*****")
    
    logger.info("\n\nStep 1: Data Pipeline")
    data_pipeline = DataPipeline(DataPipelineConfig)
    data_pipeline.run_pipeline()
    
    logger.info("\n\nStep 2: Training Pipeline")
    training_pipeline = TrainingPipeline(TrainingPipelineConfig)
    training_pipeline.run_pipeline()


if __name__ == "__main__":
    main()