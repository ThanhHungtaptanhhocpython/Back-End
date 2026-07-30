"""Unit tests for Phase 5 Task 1: BLIP-VQA Integration."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.services.reranker_service import RerankerService

class TestRerankerService(unittest.TestCase):
    
    @patch("src.services.reranker_service.Image")
    @patch("src.services.reranker_service.RerankerService._load_model")
    def test_score_image(self, mock_load, mock_image):
        service = RerankerService()
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.tokenizer.encode.side_effect = lambda x, **kwargs: [100] if x == "yes" else [200]
        service._processor = mock_processor
        
        # Mock model and logits
        # Let's say yes_logit (idx 100) = 5.0, no_logit (idx 200) = 2.0
        # Expected prob: e^5 / (e^5 + e^2) = e^3 / (e^3 + 1) = 20.085 / 21.085 ≈ 0.952
        mock_logits = MagicMock()
        
        def mock_getitem(idx):
            if idx == 100:
                mock_item = MagicMock()
                mock_item.item.return_value = 5.0
                return mock_item
            elif idx == 200:
                mock_item = MagicMock()
                mock_item.item.return_value = 2.0
                return mock_item
            return MagicMock()
            
        mock_logits.__getitem__.side_effect = mock_getitem
        
        mock_scores = [[mock_logits]]
        
        mock_outputs = MagicMock()
        mock_outputs.scores = mock_scores
        
        mock_model = MagicMock()
        mock_model.generate.return_value = mock_outputs
        service._model = mock_model
        
        # Run function
        prob = service.score_image("dummy.jpg", "Is there a dog?")
        
        # Check that prob is correct
        self.assertAlmostEqual(prob, 0.952574, places=4)
        
        # Verify model was called with output_scores=True
        mock_model.generate.assert_called_once()
        _, kwargs = mock_model.generate.call_args
        self.assertTrue(kwargs.get("output_scores"))
        self.assertTrue(kwargs.get("return_dict_in_generate"))

if __name__ == "__main__":
    unittest.main()
