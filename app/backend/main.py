import sys
import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", 
                   "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

inferencer = ModelInferencer(
    model_path=config.training_pipeline_config.artefacts_config.artefacts_path,
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG/JPEG files are allowed"
        )
    contents = await file.read()
    try:
        predictions = await inferencer.predict(contents)
        return {
            "filename": file.filename, 
            "predictions": predictions, 
            "error": None
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format for {file.filename}"
            )
        contents = await file.read()
        try:
            predictions = await inferencer.predict(contents)
            results.append({
                "filename": file.filename,
                "predictions": predictions,
                "error": None
            })
        except Exception as e:
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 