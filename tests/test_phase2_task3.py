"""Unit tests for Phase 2 Task 3: ElasticProcessor utility."""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.elastic_processing import ElasticProcessor


class TestElasticProcessor(unittest.TestCase):
    """Test the ElasticProcessor functionality by mocking the official client."""

    @patch("src.utils.elastic_processing.Elasticsearch")
    def setUp(self, mock_es_class: MagicMock) -> None:
        self.mock_es_instance = MagicMock()
        mock_es_class.return_value = self.mock_es_instance
        self.processor = ElasticProcessor("http://fake-url:9200")

    def test_create_indices(self) -> None:
        """Test index creation logic."""
        mappings = {
            "aic_ocr": {"mappings": {"properties": {"ocr_text": {"type": "text"}}}},
            "aic_asr": {"mappings": {"properties": {"text": {"type": "text"}}}}
        }
        
        # Simulate that aic_ocr doesn't exist, but aic_asr DOES exist
        def exists_side_effect(index: str) -> bool:
            return index == "aic_asr"
            
        self.mock_es_instance.indices.exists.side_effect = exists_side_effect
        
        self.processor.create_indices(mappings)
        
        # It should have checked both
        self.assertEqual(self.mock_es_instance.indices.exists.call_count, 2)
        
        # It should have only created aic_ocr
        self.mock_es_instance.indices.create.assert_called_once_with(
            index="aic_ocr", 
            body=mappings["aic_ocr"]
        )

    @patch("src.utils.elastic_processing.bulk")
    def test_bulk_index_ocr(self, mock_bulk: MagicMock) -> None:
        """Test the bulk indexing formatting."""
        mock_bulk.return_value = (2, [])  # (success_count, errors)
        
        docs = [{"ocr_text": "hello"}, {"ocr_text": "world"}]
        success = self.processor.bulk_index_ocr(docs)
        
        self.assertEqual(success, 2)
        mock_bulk.assert_called_once()
        
        # Verify the actions list passed to bulk
        actions = mock_bulk.call_args[0][1]
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["_index"], "aic_ocr")
        self.assertEqual(actions[0]["_source"], docs[0])

    def test_search_ocr(self) -> None:
        """Test search query building and response parsing."""
        fake_response = {
            "hits": {
                "hits": [
                    {
                        "_score": 1.25,
                        "_source": {"faiss_id": 100, "ocr_text": "hello test"}
                    }
                ]
            }
        }
        self.mock_es_instance.search.return_value = fake_response
        
        results = self.processor.search_ocr("hello", topk=5)
        
        # Verify search was called with correct index and size
        self.mock_es_instance.search.assert_called_once()
        kwargs = self.mock_es_instance.search.call_args[1]
        self.assertEqual(kwargs["index"], "aic_ocr")
        self.assertEqual(kwargs["body"]["size"], 5)
        
        # Verify results parsing injected the _score
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["faiss_id"], 100)
        self.assertEqual(results[0]["_score"], 1.25)


if __name__ == "__main__":
    unittest.main()
