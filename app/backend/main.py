from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import List
import numpy as np
from PIL import Image
import io
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from CNNClassifier.components.model import NNModel
from CNNClassifier.config import ModelTrainingConfig

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = NNModel()
model_path = os.path.join(os.path.dirname(__file__), '../../artefacts/model/model.pth')
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    # Resize image to match model input size
    image = image.resize((224, 224))
    # Convert to tensor and normalize
    image = torch.from_numpy(np.array(image)).float()
    image = image.permute(2, 0, 1)  # Change from HWC to CHW
    image = image / 255.0  # Normalize to [0, 1]
    return image.unsqueeze(0)  # Add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_tensor = preprocess_image(contents)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities[0].numpy()
    
    # Convert probabilities to dictionary with class names
    class_names = ModelTrainingConfig.classes
    result = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
    
    return result

@app.post("/predict_bulk")
async def predict_bulk(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        image_tensor = preprocess_image(contents)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].numpy()
        
        # Convert probabilities to dictionary with class names
        class_names = ModelTrainingConfig.classes
        result = {
            "filename": file.filename,
            "predictions": {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
        }
        results.append(result)
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 