"""Unit tests for Phase 4 Task 3: Multimodal API."""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from main import app

class TestMultimodalAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.utils.nlp_processing.QueryPlanner.parse_query")
    @patch("src.services.fusion_service.multimodal_search")
    def test_multimodal_endpoint(self, mock_search, mock_parse):
        # Mock the planner output
        mock_parse.return_value = {
            "visual_query": "Cảnh sát",
            "ocr_query": "Police",
            "asr_query": "",
            "weights": {"visual": 0.5, "ocr": 0.5, "asr": 0.0}
        }
        
        # Mock the fusion search output
        mock_search.return_value = [
            {
                "faiss_id": 101, 
                "final_score": 0.85, 
                "score_breakdown": {"visual": 0.9, "ocr": 0.8, "asr": 0.0}
            }
        ]

        response = self.client.post(
            "/users/multimodalsearch",
            json={"query": 'Cảnh sát "Police"', "topk": 50}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_items"], 1)
        self.assertEqual(data["data"]["items"][0]["faiss_id"], 101)
        self.assertEqual(data["data"]["items"][0]["final_score"], 0.85)
        
        # Verify the service was called correctly with the planner's outputs
        mock_search.assert_called_once_with(
            visual_query="Cảnh sát",
            ocr_query="Police",
            asr_query="",
            weights={"visual": 0.5, "ocr": 0.5, "asr": 0.0},
            topk=50,
            original_query='Cảnh sát "Police"',
        )

if __name__ == "__main__":
    unittest.main()
