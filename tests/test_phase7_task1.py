"""Unit tests for Phase 7 Task 1: Dual Embedding Models."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.siglip_processing import SigLIPProcessor
from src.utils.beit3_processing import BEiT3Processor

class TestDualEmbedding(unittest.TestCase):
    
    @patch("src.utils.siglip_processing.SigLIPProcessor._load_model")
    def test_siglip_feature_extraction(self, mock_load):
        processor = SigLIPProcessor()
        
        # Mock PyTorch behavior
        mock_inputs = MagicMock()
        mock_processor_instance = MagicMock(return_value=mock_inputs)
        processor._processor = mock_processor_instance
        
        mock_tensor = MagicMock()
        mock_tensor.norm.return_value = 1.0
        mock_tensor.__truediv__.return_value = mock_tensor
        mock_tensor.squeeze.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        # Return a fake 768-dim array
        mock_tensor.numpy.return_value = np.zeros(768, dtype=np.float32)
        
        mock_model = MagicMock()
        mock_model.get_text_features.return_value = mock_tensor
        processor._model = mock_model
        
        # Call method
        features = processor.get_text_features("Test query")
        
        self.assertEqual(features.shape, (768,))
        self.assertEqual(features.dtype, np.float32)
        mock_model.get_text_features.assert_called_once_with(**mock_inputs)
        
    @patch("src.utils.beit3_processing.BEiT3Processor._load_model")
    def test_beit3_feature_extraction(self, mock_load):
        processor = BEiT3Processor()
        
        mock_inputs = MagicMock()
        mock_tokenizer = MagicMock(return_value=mock_inputs)
        processor._tokenizer = mock_tokenizer
        
        mock_tensor = MagicMock()
        mock_tensor.norm.return_value = 1.0
        mock_tensor.__truediv__.return_value = mock_tensor
        mock_tensor.squeeze.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.zeros(768, dtype=np.float32)
        
        mock_outputs = MagicMock()
        mock_outputs.pooler_output = mock_tensor
        
        mock_model = MagicMock(return_value=mock_outputs)
        processor._model = mock_model
        
        # Call method
        features = processor.get_text_features("Test query")
        
        self.assertEqual(features.shape, (768,))
        self.assertEqual(features.dtype, np.float32)
        mock_model.assert_called_once_with(**mock_inputs)

if __name__ == "__main__":
    unittest.main()
