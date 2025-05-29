import sys
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
from io import BytesIO
from app.logger import logger
from app.backend.datamodels import ClassPrediction, PredictionResult, PredictionResponse, ErrorResponse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from CNNClassifier.components.model_inferencer import ModelInferencer
from CNNClassifier.config import ConfigManager


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg"}
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
logger.info("CORS Middleware Added")

inferencer = ModelInferencer(model_path=config.training_pipeline_config.artefacts_config.artefacts_path / "model_metadata.pth",     
                             device='cpu')

def validate_image_file(file: UploadFile) -> None:
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE/(1024*1024)}MB"
        )
    
    content_type = file.content_type
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {content_type}. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG/JPEG files are allowed"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a single chicken fecal image",
    description="""
    Upload a chicken fecal image (JPG/JPEG) for disease identification.
    The image should be a clear photo of chicken feces for accurate classification.
    """,
    responses={
        200: {"description": "Successful identification"},
        400: {"model": ErrorResponse, "description": "Invalid request or file format"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict(
    file: UploadFile = File(..., description="Chicken fecal image (JPG/JPEG)")
) -> PredictionResponse:
    """
    Predict the disease from a single chicken fecal image.
    
    Args:
        file: Image file to analyze (JPG/JPEG, max 10MB)
        
    Returns:
        Prediction results including class probabilities
    """
    try:
        validate_image_file(file)
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "detail": "Uploaded file is empty", 
                    "error_type": "empty_file"
                    }
            )
        logger.info(f"Processing image: {file.filename} ({len(contents)} bytes)")
        image = Image.open(BytesIO(contents))
        predicted_class, confidence, probabilities_dict = inferencer.predict(image)
        logger.info(
            f"Prediction successful - {file.filename}: "
            f"{predicted_class} (confidence: {confidence:.2f})"
            f"Probabilities: {probabilities_dict}"
        )
        return PredictionResponse(
            filename=file.filename,
            predictions={
                "label": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities_dict
            },
            error=None
        )
    except HTTPException as he:
        raise HTTPException(
            status_code=he.status_code,
            detail={
                "detail": str(he.detail),
                "error_type": "validation_error"
            }
        )
    except ValueError as ve:
        logger.error(f"Invalid image format for {file.filename}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "detail": f"Invalid image format: {str(ve)}",
                "error_type": "invalid_image"
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error processing {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": f"An unexpected error occurred while processing the image: {str(e)}",
                "error_type": "internal_error"
            }
        )

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Chicken Disease Classification API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 