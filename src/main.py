from CNNClassifier.logger import logger
from CNNClassifier.pipelines.data_pipeline import DataPipeline
from CNNClassifier.pipelines.training_pipeline import TrainingPipeline
from CNNClassifier.config import ConfigManager



def main() -> None:
    logger.info("\n\n*****\nNEW RUN\n*****")
    
    config = ConfigManager().get_config()
    logger.info("\n\nLoaded Configs\n\n")
    
    logger.info("\n\nStep 1: Data Pipeline")
    data_pipeline = DataPipeline(config.data_pipeline_config)
    # data_pipeline.run_pipeline()
    
    logger.info("\n\nStep 2: Training Pipeline")
    training_pipeline = TrainingPipeline(config.training_pipeline_config)
    training_pipeline.run_pipeline()


if __name__ == "__main__":
    main()