import sys
import os
from loguru import logger
from CNNClassifier.config import ConfigManager


log_path = ConfigManager().get_config().logs_config.logs_folder_path
os.makedirs(log_path, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(f"{log_path}/app_logs.log", level="INFO")