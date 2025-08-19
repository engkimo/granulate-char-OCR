import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import base64
from io import BytesIO
from PIL import Image
from backend.main import app
from backend.domain.entities.character import Character
from backend.domain.entities.ocr_result import OCRResult


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image():
    # Create a simple test image
    img = Image.new("RGB", (100, 50), color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestOCRAPI:
    def test_process_image_success(self, client, sample_image):
        with patch("backend.api.endpoints.ocr.OCRService") as mock_ocr_service:
            # Mock the OCR service response
            mock_service = Mock()
            mock_ocr_service.return_value = mock_service

            mock_result = OCRResult(
                image_id="test_123",
                characters=[
                    Character("ᐁ", "A", 0.95),
                    Character("ᐂ", "B", 0.92),
                    Character("ᐃ", "C", 0.88),
                ],
                processing_time=0.256,
            )
            mock_service.process_image.return_value = mock_result

            # Send request
            response = client.post(
                "/api/v1/ocr/process",
                files={"file": ("test.png", sample_image, "image/png")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "ABC"
            assert data["average_confidence"] == pytest.approx(0.9166, 0.001)
            assert len(data["characters"]) == 3

    def test_process_image_base64_success(self, client, sample_image):
        with patch("backend.api.endpoints.ocr.OCRService") as mock_ocr_service:
            mock_service = Mock()
            mock_ocr_service.return_value = mock_service

            mock_result = OCRResult(
                image_id="test_123",
                characters=[Character("ᐁ", "A", 0.95)],
                processing_time=0.150,
            )
            mock_service.process_image.return_value = mock_result

            # Convert image to base64
            base64_image = base64.b64encode(sample_image).decode("utf-8")

            response = client.post(
                "/api/v1/ocr/process-base64", json={"image": base64_image}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "A"

    def test_process_image_invalid_file_type(self, client):
        response = client.post(
            "/api/v1/ocr/process",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_process_image_no_file(self, client):
        response = client.post("/api/v1/ocr/process")

        assert response.status_code == 422

    def test_process_image_empty_result(self, client, sample_image):
        with patch("backend.api.endpoints.ocr.OCRService") as mock_ocr_service:
            mock_service = Mock()
            mock_ocr_service.return_value = mock_service

            mock_result = OCRResult(
                image_id="test_123", characters=[], processing_time=0.100
            )
            mock_service.process_image.return_value = mock_result

            response = client.post(
                "/api/v1/ocr/process",
                files={"file": ("test.png", sample_image, "image/png")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == ""
            assert data["characters"] == []

    def test_health_check(self, client):
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
