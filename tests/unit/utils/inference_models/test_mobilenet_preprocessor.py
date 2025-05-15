import unittest
import numpy as np
import io
from PIL import Image
from unittest.mock import patch, MagicMock

from app.utils.inference_models.mobilenet_preprocessor import MobileNetProcessor


class TestMobileNetInputPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = MobileNetProcessor()

    def test_preprocess_input_shape(self):
        """Test that the output has the correct shape"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Process the image
        result = self.preprocessor.preprocess_input(img_bytes)

        # Check shape is correct (batch_size=1, height=224, width=224, channels=3)
        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_preprocess_input_normalization(self):
        """Test that the output values are properly normalized"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Process the image
        result = self.preprocessor.preprocess_input(img_bytes)

        # Check values are in the expected range for MobileNetV2 preprocessing
        # MobileNetV2 uses values in [-1, 1] range
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    @patch('app.utils.inference_models.mobilenet_preprocessor.image.load_img')
    @patch('app.utils.inference_models.mobilenet_preprocessor.img_to_array')
    @patch('app.utils.inference_models.mobilenet_preprocessor.preprocess_input')
    def test_preprocess_input_flow(self, mock_preprocess, mock_to_array, mock_load_img):
        """Test the function flow with mocks"""
        # Setup mocks
        mock_img = MagicMock()
        mock_load_img.return_value = mock_img

        mock_array = np.zeros((224, 224, 3))
        mock_to_array.return_value = mock_array

        expected_result = np.zeros((1, 224, 224, 3))
        mock_preprocess.return_value = expected_result

        # Test
        test_bytes = b'test image bytes'
        result = self.preprocessor.preprocess_input(test_bytes)

        # Verify
        mock_load_img.assert_called_once()
        mock_to_array.assert_called_once_with(mock_img)
        # Check that preprocess_input was called with expanded dimensions
        self.assertEqual(
            mock_preprocess.call_args[0][0].shape, (1, 224, 224, 3))
        self.assertEqual(result.all(), expected_result.all())


if __name__ == '__main__':
    unittest.main()
