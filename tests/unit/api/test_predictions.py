import io
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import UploadFile, HTTPException
from fastapi.testclient import TestClient

from app.core.errors import InvalidImageError, ModelNotLoadedError
from app.main import app
from app.services.general_classifier import GeneralClassifierService
from app.services.apolo_classifier import ApoloClassifierService

client = TestClient(app)

# Sample test image path - we'll use this to load a test image
TEST_IMAGE_PATH = os.path.join(os.path.dirname(
    __file__), "../test_data/test_dog.jpg")

# Create test image directory if it doesn't exist
os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)


@pytest.fixture
def mock_image():
    """
    Create a mock image for testing or use an existing one if available
    """
    # If test image doesn't exist, create a blank one
    if not os.path.exists(TEST_IMAGE_PATH):
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='white')
        img.save(TEST_IMAGE_PATH)

    # Return the file path
    return TEST_IMAGE_PATH


@pytest.fixture
def mock_apolo_classifier():
    mock = MagicMock(spec=ApoloClassifierService)
    with patch("app.api.routes.predictions.get_apolo_classifier_service", return_value=mock):
        yield mock


@pytest.fixture
def valid_image_file(mock_image):
    """Fixture that returns a valid image file for testing API endpoints"""
    # Open the real test image file and read its content
    with open(mock_image, "rb") as img_file:
        image_content = img_file.read()

    # Return dictionary with image file in the format expected by FastAPI's TestClient
    return {"image": ("test.jpg", io.BytesIO(image_content), "image/jpeg")}


@pytest.fixture
def invalid_image_file():
    return {"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}


class TestClassifyEndpoint:
    @patch('app.api.dependencies.general_classifier_service')
    def test_classify_success(self, mock_service, valid_image_file):
        # Set up mock return value
        mock_service.predict.return_value = {
            "dog": 0.8,
            "cat": 0.15,
            "other": 0.05
        }

        # Make the request
        response = client.post("/api/v1/classify", files=valid_image_file)

        # Assert response
        assert response.status_code == 200
        assert response.json()["top_prediction"] == "dog"
        assert len(response.json()["predictions"]) == 3
        assert "processing_time" in response.json()

        # Verify mock was called correctly
        mock_service.predict.assert_called_once()

    @patch('app.api.dependencies.general_classifier_service')
    def test_classify_invalid_image(self, mock_classifier_service, invalid_image_file):
        response = client.post("/api/v1/classify", files=invalid_image_file)

        assert response.status_code == 400
        assert "Invalid image format" in response.json()["detail"]
        mock_classifier_service.predict.assert_not_called()

    @patch('app.api.dependencies.general_classifier_service')
    def test_classify_model_not_loaded(self, mock_general_classifier, valid_image_file):
        # Make the mock raise an exception
        mock_general_classifier.predict.side_effect = ModelNotLoadedError()

        response = client.post("/api/v1/classify", files=valid_image_file)

        assert response.status_code == 503
        assert "Model is not loaded" in response.json()["detail"]

    @patch('app.api.dependencies.general_classifier_service')
    def test_classify_general_exception(self, mock_general_classifier, valid_image_file):
        # Make the mock raise a general exception
        mock_general_classifier.predict.side_effect = Exception(
            "Unexpected error")

        response = client.post("/api/v1/classify", files=valid_image_file)

        assert response.status_code == 500
        assert "An unexpected error occurred" in response.json()["detail"]
