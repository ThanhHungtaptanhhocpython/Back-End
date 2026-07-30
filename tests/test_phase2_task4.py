"""Unit tests for Phase 2 Task 4: Elasticsearch API Endpoints."""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from main import app

class TestElasticAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.services.user_service.get_elastic_processor")
    def test_ocr_search_endpoint(self, mock_get_processor):
        mock_processor = MagicMock()
        mock_processor.search_ocr.return_value = [{"faiss_id": 101, "ocr_text": "Sale 50%", "_score": 1.5}]
        mock_get_processor.return_value = mock_processor

        response = self.client.post(
            "/users/ocrsearch",
            json={"query": "Sale", "topk": 10}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_items"], 1)
        self.assertEqual(data["data"]["items"][0]["faiss_id"], 101)
        
        # Verify the service was called correctly
        mock_processor.search_ocr.assert_called_once_with("Sale", topk=10)

    @patch("src.services.user_service.get_elastic_processor")
    def test_asr_search_endpoint(self, mock_get_processor):
        mock_processor = MagicMock()
        mock_processor.search_asr.return_value = [{"faiss_id": 202, "text": "Hello world", "_score": 2.1}]
        mock_get_processor.return_value = mock_processor

        response = self.client.post(
            "/users/asrsearch",
            json={"query": "Hello", "topk": 5}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_items"], 1)
        self.assertEqual(data["data"]["items"][0]["faiss_id"], 202)
        
        # Verify the service was called correctly
        mock_processor.search_asr.assert_called_once_with("Hello", topk=5)

    @patch("src.services.user_service.get_elastic_processor")
    def test_ocrandodsearch_endpoint_alias(self, mock_get_processor):
        # This endpoint should act as an alias to OCR search
        mock_processor = MagicMock()
        mock_processor.search_ocr.return_value = [{"faiss_id": 303, "ocr_text": "Legacy UI request"}]
        mock_get_processor.return_value = mock_processor

        response = self.client.post(
            "/users/ocrandodsearch",
            json={"query": "Legacy", "topk": 100}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["items"][0]["faiss_id"], 303)


if __name__ == "__main__":
    unittest.main()
