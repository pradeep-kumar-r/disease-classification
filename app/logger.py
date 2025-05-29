import sys
import os
from loguru import logger


log_path = "../logs/"
os.makedirs(log_path, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(f"{log_path}/app_logs.log", level="INFO")