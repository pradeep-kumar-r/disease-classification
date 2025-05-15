from CNNClassifier.logger import logger
from CNNClassifier.pipelines.data_pipeline import DataPipeline
# from CNNClassifier.pipelines.training_pipeline import DataPipeline


def main() -> None:
      logger.info("\n\n*****\nNEW RUN\n\n")
      
      # Step 1: Data Pipeline
      logger.info("Starting Data Pipeline")
      data_pipeline = DataPipeline()
      data_pipeline.run_pipeline()
      logger.info("Data Pipeline completed")



if __name__ == "__main__":
      main()