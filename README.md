# Chicken Disease Classification from Fecal Images

This repository provides an end-to-end solution for identifying chicken diseases, specifically Coccidiosis, Salmonella, and Newcastle Disease from fecal images using deep learning. It includes a modular training pipeline and a web application for inference.

## Overview

The project aims to demonstrate best practices in structuring machine learning projects and building simple web applications for model deployment. It caters to Data Scientists and ML Engineers looking to understand modular code design for ML pipelines and web app integration.

## Key Features

*   **Modular Training Pipeline:**
    *   Automated data download from Kaggle API.
    *   Configurable data transformation pipeline.
    *   Flexible model training and evaluation modules.
*   **Web Application for Inference:**
    *   FastAPI backend for model serving.
    *   Streamlit frontend for user interaction (image upload and prediction display).
    *   Dockerized setup for easy deployment.
*   **Educational Focus:** Designed to illustrate software engineering best practices in an ML context.

## Project Goal

This repository is primarily for educational and learning purposes. It aims to guide Data Scientists and Machine Learning Engineers on:

*   Writing modular and maintainable code for ML projects.
*   Implementing end-to-end ML pipelines.
*   Building and deploying simple web applications for ML model inference.
*   Utilizing tools like Docker for containerization.

## Directory Structure

```
├── app/                    # Contains Dockerfile and code for FE (Streamlit) and BE (FastAPI)
│   ├── backend/
│   └── frontend/
├── config/                 # Configuration files (e.g., config.yaml)
├── data/                   # Data (typically gitignored, populated by pipeline)
├── models/                 # Trained models (typically gitignored, populated by pipeline)
├── notebooks/              # Jupyter notebooks for experimentation
├── research/               # Research artifacts
├── src/                    # Source code
│   └── CNNClassifier/      # Core ML pipeline code (data ingestion, model training, etc.)
│       ├── __init__.py
│       ├── components/     # Individual pipeline components
│       ├── config/         # Configuration management
│       ├── constants/      # Project constants
│       ├── entity/         # Entity definitions
│       ├── pipeline/       # Training and prediction pipelines
│       ├── utils/          # Utility functions
│       └── main.py         # Main script to run the training pipeline
├── main.py                 # Main script to run the training pipeline (if different from src/CNNClassifier/main.py)
├── requirements.txt        # Python dependencies for the ML pipeline
├── setup.py                # Package setup script
├── template.py             # Script to generate project structure
└── README.md               # This file
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   Docker & Docker Compose
*   Kaggle Account & API Token (for training with fresh data download)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/r-pradeep-kumar/disease-classification.git # Replace with your repo URL
    cd disease-classification
    ```

2.  **Set up Python Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### Environment Variables

This project requires certain environment variables to be set, primarily for accessing the Kaggle API to download the dataset. These variables should be stored in a `.env` file in the root directory of the project.

Create a file named `.env` and add the following, replacing the placeholder values with your actual Kaggle credentials:

```env
KAGGLE_USER="your_kaggle_username"
KAGGLE_KEY="your_kaggle_api_key"
```

**Note:** The `.env` file is included in `.gitignore` to prevent accidental commitment of sensitive credentials. Ensure you have this file set up locally if you intend to run the data ingestion part of the pipeline.

## Usage

There are two main ways to use this repository:

### 1. Training a Custom Model / Modifying the Pipeline

This option allows you to modify the existing pipeline, add more data, change model architectures, or adjust training parameters.

**a. Configure Kaggle API:**

   To enable automatic data download, you need to set up your Kaggle API credentials.
   1.  Go to your Kaggle account page, and click 'Create New API Token'. This will download `kaggle.json`.
   2.  Place the `kaggle.json` file in the appropriate location (e.g., `~/.kaggle/kaggle.json` on Linux/macOS or `C:\Users\<Your-Username>\.kaggle\kaggle.json` on Windows).
   3.  Ensure the dataset URL or command in `src/CNNClassifier/components/data_ingestion.py` (or your configuration file) matches the desired Kaggle dataset.

**b. (Optional) Modify Configurations:**

   Adjust parameters in `config/config.yaml` and `params.yaml` (if used) to change aspects like learning rate, batch size, epochs, model architecture details, or data paths.

**c. (Optional) Modify Code:**

   You can modify the Python scripts within the `src/CNNClassifier/` directory to:
   *   Implement new data augmentation techniques.
   *   Try different model architectures.
   *   Customize the training loop or evaluation metrics.

**d. Run the Training Pipeline:**

   Execute the main script to run the entire pipeline (data ingestion, preparation, model training, evaluation):
   ```bash
   python src/CNNClassifier/main.py
   ```

### 2. Running the Web App with the Pre-trained Model

This option allows you to quickly run the web application using a pre-trained model (assuming one is provided or has been generated by a previous training run and is correctly referenced by the backend).

**a. Ensure Docker is running.**

**b. Build and Run Docker Containers:**

   From the root directory of the project, run:
   ```bash
   docker-compose up --build
   ```
   This command will build the images for the frontend (Streamlit) and backend (FastAPI) services and then start the containers.

**c. Access the Web App:**

   *   **Frontend (Streamlit):** Open your web browser and go to `http://localhost:8501`
   *   **Backend (FastAPI Docs):** You can explore the API at `http://localhost:8000/docs`

   You can now upload a chicken fecal image through the Streamlit interface and get a prediction from the deployed model.

## Feedback and Contributions

Feedback on how to refactor the codebase, improve modularity, or streamline any part of the project is always welcome!

If you'd like to contribute, please feel free to fork the repository, make your changes, and submit a pull request.

---