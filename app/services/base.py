from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Union
from io import BytesIO
from PIL import Image

from app.core.errors import InvalidImageError
from app.utils.inference_models.preprocessor import InputPreprocessor


class BaseClassifierService(ABC):
    """Base class for all classifier services"""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_path = None

    def load_model(self):
        """Load model from local file or S3 (placeholder)"""
        # This will be implemented when we add TensorFlow
        # For now, we'll just set a flag to indicate the model is loaded
        self.model_loaded = True

    def preprocess_image(self, image_data: bytes) -> Any:
        """
        Process the image data into the format required by the model

        Args:
            image_data: Raw bytes of the uploaded image

        Returns:
            Processed image ready for inference

        Raises:
            InvalidImageError: If the image cannot be processed
        """
        try:
            # Open image using PIL
            img = Image.open(BytesIO(image_data))

            # For now, just convert to RGB and resize to a standard size
            # This will be enhanced when we add actual model implementation
            img = img.convert('RGB')
            img = img.resize((224, 224))

            # In real implementation, we would convert to numpy array or tensor

            return img
        except Exception as e:
            raise InvalidImageError(f"Failed to process image: {str(e)}")

    @abstractmethod
    def predict(self, image_data: bytes) -> Union[Dict[str, float], Tuple[str, float]]:
        """
        Process image and run inference

        Args:
            image_data: Raw bytes of the uploaded image

        Returns:
            Either a dictionary of class probabilities or a tuple of (result, confidence)

        Raises:
            ModelNotLoadedError: If the model is not loaded
            InvalidImageError: If the image cannot be processed
        """
        pass
