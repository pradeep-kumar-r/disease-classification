from CNNClassifier.logger import logger
from CNNClassifier.pipelines.data_pipeline import DataPipeline
from CNNClassifier.config import DataPipelineConfig
# from CNNClassifier.pipelines.training_pipeline import DataPipeline


def main() -> None:
      logger.info("\n\n*****\nNEW RUN\n\n")
      
      # Step 1: Data Pipeline
      data_pipeline = DataPipeline(DataPipelineConfig)
      data_pipeline.run_pipeline()
      


if __name__ == "__main__":
      main()