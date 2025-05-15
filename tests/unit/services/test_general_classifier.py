import unittest
from unittest.mock import patch, MagicMock
import os

from app.services.general_classifier import GeneralClassifierService


class TestGeneralClassifierService(unittest.TestCase):
    """Test cases for the GeneralClassifierService"""

    def setUp(self):
        """Set up for each test case"""
        # Load the sample test image
        test_image_path = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'test_data', 'test_dog.jpg')
        with open(test_image_path, 'rb') as f:
            self.sample_image_data = f.read()

    @patch('app.services.general_classifier.config')
    @patch('app.utils.inference_models.model_loader_manager.config')
    @patch('app.utils.inference_models.model_loader_manager.ModelLoaderManager.get_loader')
    def test_initialization(
            self,
            mock_get_loader: MagicMock,
            mock_config_model_loader_manager: MagicMock,
            mock_config_general_classifier: MagicMock):
        """Test that the service initializes correctly"""
        # Configure mocks
        mock_config_model_loader_manager.MODEL_SOURCE = "local"
        mock_config_general_classifier.CAT_DOG_OTHER_CLASSIFIER = "test_model.h5"
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.load.return_value = mock_model

        # Initialize the service with mocks in place
        service = GeneralClassifierService()

        # Verify initialization
        self.assertEqual(service.model_filename, "test_model.h5")
        self.assertEqual(service.model, mock_model)
        mock_loader.load.assert_called_once_with("test_model.h5")

    @patch('os.getenv')
    @patch('app.utils.inference_models.model_loader_manager.ModelLoaderManager.get_loader')
    def test_predict_success(self, mock_get_loader: MagicMock, mock_getenv: MagicMock):
        """Test successful prediction"""
        # Configure mocks
        mock_getenv.return_value = "test_model.h5"
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.load.return_value = mock_model

        # Initialize the service with mocks in place
        service = GeneralClassifierService()

        # Instead of mocking preprocess_image, let it process the real image
        # Configure mock model to return fake prediction results
        mock_result = MagicMock()
        mock_result.shape = (1, 3)  # Assuming 3 classes

        # Create a mock prediction array representing class probabilities
        import numpy as np
        # Dog: 70%, Cat: 10%, etc.
        mock_prediction = np.array([[0.7, 0.2, 0.1]])
        mock_model.predict.return_value = mock_prediction

        # Call predict and verify results
        result = service.predict(self.sample_image_data)

        # Verify the result format
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(k, str) for k in result.keys()))
        self.assertTrue(all(isinstance(v, float) for v in result.values()))

        # Check that we have the expected classes
        expected_classes = ["dog", "cat", "other"]
        for cls in expected_classes:
            self.assertIn(cls, result)

        # Verify probabilities sum to approximately 1.0
        self.assertAlmostEqual(sum(result.values()), 1.0, places=7)


if __name__ == '__main__':
    unittest.main()
