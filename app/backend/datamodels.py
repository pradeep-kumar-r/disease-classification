from pydantic import BaseModel, Field, confloat
from typing import Optional, Dict


class ClassPrediction(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence score (0-1)")


class PredictionResult(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution over all classes")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing what went wrong")
    error_type: str = Field(..., description="Type of error that occurred")


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Original filename of the uploaded image")
    predictions: Optional[PredictionResult] = Field(None, description="Prediction results")
    error: Optional[ErrorResponse] = Field(None, description="Error details if prediction failed")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "chicken_image.jpg",
                "predictions": {
                    "label": "Salmonella",
                    "confidence": 0.95,
                    "probabilities": {
                        "Salmonella": 0.95,
                        "New Castle Disease": 0.03,
                        "Coccidiosis": 0.01,
                        "Healthy": 0.01
                    }
                },
                "error": None
            }
        }