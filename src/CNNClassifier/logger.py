import sys
import os
from loguru import logger
from CNNClassifier.config import LogsConfig


log_path = LogsConfig.log_folder_path
os.makedirs(log_path, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(f"{log_path}/debug_logs.log", level="DEBUG")
logger.add(f"{log_path}/logs.log", level="INFO")


# Test the logger
if __name__ == "__main__":
    logger.info("Testing whether logging works correctly")
    logger.debug("DEBUG - Testing whether logging works correctly")
