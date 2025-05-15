from loguru import logger
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io

from app.utils.inference_models.preprocessor import InputPreprocessor


class MobileNetProcessor(InputPreprocessor):

    def __init__(self):
        pass

    def preprocess_input(self, input: bytes):
        # Convert bytes to image
        img = image.load_img(io.BytesIO(input), target_size=(224, 224))

        # Convert image to array
        img_array = img_to_array(img)

        # Expand dimensions to create batch dimension (batch size of 1)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply MobileNetV2 specific preprocessing (scaling, normalization)
        return preprocess_input(img_array)
