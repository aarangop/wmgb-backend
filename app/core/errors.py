from fastapi import HTTPException, status
from loguru import logger
from pydantic import ValidationError


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ModelInferenceError(Exception):
    """Custom exception for model inference errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ModelNotLoadedError(Exception):
    """Raised when a model is not loaded but an inference is attempted"""
    pass


class InvalidImageError(Exception):
    """Raised when an image cannot be processed"""
    pass


def handle_image_processing_error(error: ImageProcessingError):
    """Handle image processing errors."""
    return {"error": "Image processing failed", "details": str(error)}


def handle_model_inference_error(error: ModelInferenceError):
    """Handle model inference errors."""
    return {"error": "Model inference failed", "details": str(error)}


def model_not_loaded_exception():
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model is not loaded"
    )


def invalid_image_exception():
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid image format"
    )


def general_error_exception(e):
    logger.error(str(e))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred. Please try again later."
    )


def validation_error_exception(e: ValidationError):
    errors = "\n".join(e.errors(include_context=True))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=errors
    )
