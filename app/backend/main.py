import sys
import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from app.logger import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from CNNClassifier.components.model_inferencer import ModelInferencer
from CNNClassifier.config import ConfigManager


config = ConfigManager().get_config()

app = FastAPI(
    title="Chicken Disease Classification API",
    description="API for classifying whether 3 different types of chicken diseases are present or not from fecal images using deep learning",
    version="1.0.0",
    contact={
        "name": "Pradeep",
        "github": "https://github.com/pradeep-kumar-r"
    },
    license_info={
        "name": "Apache 2.0",
    }
)
logger.info("App Created")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", 
                   "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
logger.info("CORS Middleware Added")

inferencer = ModelInferencer(
    model_path=config.training_pipeline_config.artefacts_config.artefacts_path / "model_metadata.pth",
    device='cpu'
)

class PredictionResponse(BaseModel):
    filename: Optional[str]
    predictions: Optional[dict]
    error: Optional[str]
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "chest_xray.jpg",
                "predictions": {
                    "label": "Salmonella",
                    "probabilities": {
                        "Salmonella": 0.5,
                        "New Castle Disease": 0.2,
                        "Coccidiosis": 0.2,
                        "Healthy": 0.1
                    }
                },
                "error": None
            }
        }


@app.post("/predict",
         response_model=PredictionResponse,
         summary="Classify a single chicken fecal image",
         description="Upload a chicken fecal image (JPG) for disease identification",
         responses={
             200: {"description": "Successful identification"},
             400: {"description": "Invalid file format"},
             500: {"description": "Internal server error"}
         })
async def predict(file: UploadFile = File(..., description="Chicken fecal image (JPG)")
                  ) -> PredictionResponse:
    if not file.filename.lower().endswith(('.jpg', '.jpeg')):
        logger.error("Incorrect file uploaded or file not found")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG/JPEG files are allowed"
        )
    contents = await file.read()
    logger.info(f"File uploaded & read successfully: {file.filename}")
    try:
        predicted_class, confidence, probabilities_dict = await inferencer.predict(contents)
        logger.info(f"Prediction Successful for {file.filename}, predicted class: {predicted_class}, confidence: {confidence}")
        return {
            "filename": file.filename,
            "predictions": {
                "label": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities_dict
            },
            "error": None
        }
    except Exception as e:
        logger.error(f"Error during Prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        ) from e


@app.post("/predict_bulk",
         response_model=List[PredictionResponse],
         summary="Classify multiple chicken fecal images",
         description="Upload multiple chicken fecal images (JPG) for bulk idenficication",
         responses={
             200: {"description": "Successful identification"},
             400: {"description": "Invalid file format in one or more files"},
             500: {"description": "Internal server error"}
         })
async def predict_bulk(
    files: List[UploadFile] = File(..., description="List of chicken fecal images in JPG format")
) -> List[PredictionResponse]:
    results = []
    for file in files:
        if not file.filename.lower().endswith(('.jpg', '.jpeg')):
            logger.error("Incorrect file uploaded or file not found")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format for {file.filename}"
            )
        contents = await file.read()
        logger.info(f"File uploaded & read successfully: {file.filename}")
        try:
            predicted_class, confidence, probabilities_dict = await inferencer.predict(contents)
            logger.info(f"Prediction Successful for {file.filename}, predicted class: {predicted_class}, confidence: {confidence}")
            results.append({
                "filename": file.filename,
                "predictions": {
                    "label": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities_dict
                },
                "error": None
            })
        except Exception as e:
            logger.error(f"Error during Prediction of {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "predictions": None,
                "error": str(e)
            })
    return results


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Chicken Disease Classification API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 