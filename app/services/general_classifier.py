import random
from typing import Dict
import os

import numpy as np
from loguru import logger

from app.services.base import BaseClassifierService
from app.core.errors import ModelNotLoadedError
from app.utils.inference_models.mobilenet_preprocessor import MobileNetProcessor
from app.core.config import config
from app.utils.inference_models.model_repository import ModelRepository


class GeneralClassifierService(BaseClassifierService):
    """Service for general image classification"""

    def __init__(self, model_repository: ModelRepository):
        super().__init__()

        self.model_name = config.CAT_DOG_OTHER_CLASSIFIER
        self.model_repo = model_repository
        self.model = None  # Load model lazily
        self.model_processor = MobileNetProcessor()
        self.pred_classes = ['cat', 'dog', 'other']

    def predict(self, image_data: bytes) -> Dict[str, float]:
        """
        Classify an image using a pre-trained model

        Args:
            image_data: Raw bytes of the uploaded image

        Returns:
            Dictionary mapping class names to probabilities

        Raises:
            ModelNotLoadedError: If the model is not loaded
            InvalidImageError: If the image cannot be processed
        """
        if self.model is None:
            # Attempt to load the model
            try:
                self.model = self.model_repo.get_model(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelNotLoadedError(
                    "General classifier model not loaded")

        # Preprocess the image
        preprocessed_image = self.model_processor.preprocess_input(image_data)

        pred = self.model.predict(preprocessed_image)
        # For the placeholder implementation, return random classification results
        # This will be replaced with actual model inference when we add TensorFlow
        decoded_preds = {class_name: pred[0][i]
                         for i, class_name in enumerate(self.pred_classes)}

        return decoded_preds
