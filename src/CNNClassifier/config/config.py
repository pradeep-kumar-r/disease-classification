import os
from dotenv import load_dotenv
import yaml


# Load env variables
load_dotenv()

# Load config yaml file
with open("../../../config.yaml", "r") as file:
    config = yaml.safe_load(file)

data_path = config['data_folder_path']
logs_path = config['logs_folder_path']
artefacts_path = config['artefacts_folder_path']

# Check & Create the data directory
os.makedirs(data_path, exist_ok=True)
