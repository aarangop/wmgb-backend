import time
from typing import Dict, List
from fastapi import APIRouter, Depends, File, UploadFile
from loguru import logger
from pydantic import ValidationError

from app.api.dependencies import (
    get_general_classifier_service,
    get_apolo_classifier_service
)
from app.core.errors import InvalidImageError, ModelNotLoadedError, invalid_image_exception, model_not_loaded_exception, general_error_exception, validation_error_exception
from app.models.schemas import ClassificationResponse, DetailedClassificationResponse
from app.services.general_classifier import GeneralClassifierService
from app.services.apolo_classifier import ApoloClassifierService

router = APIRouter(tags=["predictions"])


@router.post("/classify", response_model=DetailedClassificationResponse)
async def classify_image(
    image: UploadFile = File(...),
    service: GeneralClassifierService = Depends(get_general_classifier_service)
):
    """
    Classify an image using the base model
    """
    try:
        start_time = time.time()
        image_data = await image.read()

        # Validate image
        if not image.content_type in ["image/jpeg", "image/png"]:
            raise InvalidImageError()

        # Process image and get predictions
        predictions = service.predict(image_data)

        # Get top prediction
        top_prediction = max(predictions.items(), key=lambda x: x[1])[0]

        # Format predictions for response
        formatted_predictions: List = [
            {"class_name": k, "probability": float(v)} for k, v in predictions.items()]

        processing_time = time.time() - start_time
        return DetailedClassificationResponse(
            predictions=formatted_predictions,
            top_prediction=top_prediction,
            processing_time=processing_time
        )

    except InvalidImageError:
        raise invalid_image_exception()
    except ModelNotLoadedError:
        raise model_not_loaded_exception()
    except ValidationError as e:
        logger.info(str(e))
        raise validation_error_exception(e)
    except Exception as e:
        raise general_error_exception(e)


@router.post("/is-apolo", response_model=ClassificationResponse)
async def is_apolo(
    image: UploadFile = File(...),
    service: ApoloClassifierService = Depends(get_apolo_classifier_service)
):
    """
    Determine if the image is of Apolo (my good boy)
    """
    try:
        start_time = time.time()
        image_data = await image.read()

        # Validate image
        if not image.content_type in ["image/jpeg", "image/png"]:
            raise invalid_image_exception()

        # Process image and get predictions
        result, confidence = service.predict(image_data)

        processing_time = time.time() - start_time

        return ClassificationResponse(
            prediction=result,
            confidence=confidence,
            processing_time=processing_time
        )

    except InvalidImageError:
        raise invalid_image_exception()
    except ModelNotLoadedError:
        raise model_not_loaded_exception()
    except Exception as e:
        raise general_error_exception(e)
