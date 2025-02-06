import os
import kaggle
from dotenv import load_dotenv
import yaml


# Load env variables & config file
load_dotenv()
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
data_path = config['data_folder_path']

# Check & Create the data directory
os.makedirs(data_path, exist_ok=True)

# Download latest version through kaggle api
try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
                dataset="allandclive/chicken-disease-1",
                path=data_path,
                unzip=True
    )
    print(f"Data downloaded successfully to {data_path}")
except Exception as e:
    print(f"Error using Kaggle API: {str(e)}")
