import sys
import os
import yaml
from loguru import logger


# Load config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
log_path = config['log_folder_path']

# Check & Create the logs directory
os.makedirs(log_path, exist_ok=True)

# Add relevant log handlers
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(f"{log_path}/debug_logs.log", level="DEBUG")
logger.add(f"{log_path}/logs.log", level="INFO")

if __name__ == "__main__":
    logger.info("Testing whether logging works correctly")
    logger.debug("DEBUG - Testing whether logging works correctly")
