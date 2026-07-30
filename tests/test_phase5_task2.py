"""Unit tests for Phase 5 Task 2: Dynamic VQA Question Formulation."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.nlp_processing import QueryPlanner

class TestVQAQuestionFormulation(unittest.TestCase):
    
    @patch("src.utils.nlp_processing.Translation.__call__")
    def test_generate_vqa_question(self, mock_translate):
        # Mock translation to pretend we translated from VN to EN
        mock_translate.return_value = "man riding a bicycle"
        
        query = "người đàn ông đi xe đạp"
        question = QueryPlanner.generate_vqa_question(query)
        
        # The result should be "Is there a man riding a bicycle?"
        self.assertEqual(question, "Is there a man riding a bicycle?")
        mock_translate.assert_called_once_with("người đàn ông đi xe đạp")

    @patch("src.utils.nlp_processing.Translation.__call__")
    def test_generate_vqa_question_with_quotes(self, mock_translate):
        # Mock translation
        mock_translate.return_value = "police car"
        
        # Should strip quotes before translation
        query = '"xe cảnh sát"'
        question = QueryPlanner.generate_vqa_question(query)
        
        self.assertEqual(question, "Is there a police car?")
        mock_translate.assert_called_once_with("xe cảnh sát")

if __name__ == "__main__":
    unittest.main()
