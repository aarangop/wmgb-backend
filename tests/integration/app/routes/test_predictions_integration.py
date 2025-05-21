# filepath: /Users/andresap/repos/whos-my-good-boy/backend/tests/integration/app/routes/test_predictions.py
import io
import os
import pytest
from fastapi.testclient import TestClient
from app.core.config import config
from app.main import app

client = TestClient(app)

# Path to the test image
TEST_IMAGE_PATH = os.path.join(
    "tests",
    "test_data",
    "test_dog.jpg"
)

api_prefix = config.API_PREFIX

# Ensure the model source is set to S3 for integration tests
config.USE_LOCAL_MODEL_REPO = False


@pytest.mark.integration
class TestPredictionEndpoints_Integration:
    """Real integration tests for the prediction endpoints"""

    def test_classify_valid_image(self):
        """Test classification with a real image using the actual model"""
        # Open and read the test image
        with open(TEST_IMAGE_PATH, "rb") as image_file:
            image_data = image_file.read()

        # Create an in-memory file-like object
        test_image = io.BytesIO(image_data)
        test_image.name = "test_dog.jpg"

        # Send request with the real test image
        response = client.post(
            f"{api_prefix}/classify",
            files={"image": ("test_dog.jpg", test_image, "image/jpeg")}
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "top_prediction" in data
        assert "predictions" in data
        assert "processing_time" in data

        # Check predictions are valid
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0

        # Each prediction should have class_name and probability
        for pred in data["predictions"]:
            assert "class_name" in pred
            assert "probability" in pred
            assert isinstance(pred["probability"], float)
            assert 0 <= pred["probability"] <= 1

    def test_classify_invalid_image_type(self):
        """Test error handling for invalid image type"""
        # Create a test file with non-image content
        test_file = io.BytesIO(b"This is not an image")

        # Send request with invalid file type
        response = client.post(
            f"{api_prefix}/classify",
            files={"image": ("test_file.txt", test_file, "text/plain")}
        )

        # Assertions
        assert response.status_code == 400
        assert "Invalid image format" in response.json()["detail"]

    def test_classify_missing_image(self):
        """Test error handling when image is missing"""
        # Send request without image
        response = client.post(f"{api_prefix}/classify", files={})

        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()
        assert "detail" in data
        assert any("image" in error["loc"] for error in data["detail"])

    @pytest.mark.skip(reason="is-apolo endpoint not fully implemented yet")
    def test_is_apolo_valid_image(self):
        """Test is-apolo classification with a real image using the actual model"""
        # Open and read the test image
        with open(TEST_IMAGE_PATH, "rb") as image_file:
            image_data = image_file.read()

        # Create an in-memory file-like object
        test_image = io.BytesIO(image_data)
        test_image.name = "test_dog.jpg"

        # Send request with the real test image
        response = client.post(
            f"{api_prefix}/is-apolo",
            files={"image": ("test_dog.jpg", test_image, "image/jpeg")}
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "prediction" in data
        assert "confidence" in data
        assert "processing_time" in data

        # Verify prediction types
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1

    @pytest.mark.skip(reason="is-apolo endpoint not fully implemented yet")
    def test_is_apolo_invalid_image_type(self):
        """Test error handling for invalid image type in is-apolo endpoint"""
        # Create a test file with non-image content
        test_file = io.BytesIO(b"This is not an image")

        # Send request with invalid file type
        response = client.post(
            f"{api_prefix}/is-apolo",
            files={"image": ("test_file.txt", test_file, "text/plain")}
        )

        # Assertions
        assert response.status_code == 400
        assert "Invalid image format" in response.json()["detail"]

    @pytest.mark.skip(reason="is-apolo endpoint not fully implemented yet")
    def test_is_apolo_missing_image(self):
        """Test error handling when image is missing in is-apolo endpoint"""
        # Send request without image
        response = client.post(f"{api_prefix}/is-apolo", files={})

        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()
        assert "detail" in data
        assert any("image" in error["loc"] for error in data["detail"])
