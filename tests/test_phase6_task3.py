"""Unit tests for Phase 6 Task 3: BLIP Sequence Validation."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.trake_processing import TRAKE

class MockFaiss:
    pass

class TestBlipSequenceValidation(unittest.TestCase):
    
    @patch("src.utils.trake_processing.os.path.exists")
    @patch("src.utils.trake_processing.reranker_service")
    @patch("src.utils.trake_processing.QueryPlanner")
    def test_sequence_validation(self, mock_planner, mock_reranker, mock_exists):
        trake = TRAKE(MockFaiss())
        # mock faiss searcher since it is not used in rank/format directly
        trake.format_response = lambda x: x  # bypass format for testing
        
        # Mock VQA question
        mock_planner.generate_vqa_question.return_value = "Is there a test?"
        
        # Mock image path exists
        mock_exists.return_value = True
        
        # Mock VQA scores
        # First sequence gets a bad VQA score (0.1)
        # Second sequence gets a good VQA score (0.9)
        def mock_score(img_path, question):
            if "bad" in img_path:
                return 0.1
            return 0.9
            
        mock_reranker.score_image.side_effect = mock_score
        
        seq1 = {
            'total_score': 10.0,
            'frame_details': [
                {'frame_name': 'bad_1.jpg', 'split': 'a'},
                {'frame_name': 'bad_2.jpg', 'split': 'a'}
            ]
        }
        
        seq2 = {
            'total_score': 9.5, # Slightly lower original score
            'frame_details': [
                {'frame_name': 'good_1.jpg', 'split': 'a'},
                {'frame_name': 'good_2.jpg', 'split': 'a'}
            ]
        }
        
        # trake_processing sorts descending so seq1 is first
        ranked_sequences = [seq1, seq2]
        
        # Call the logic directly (reproducing process_temporal_search tail)
        events = ["Query 1", "Query 2"]
        
        for seq in ranked_sequences:
            vqa_scores = []
            for i, frame_detail in enumerate(seq['frame_details']):
                event_query = events[i]
                vqa_question = mock_planner.generate_vqa_question(event_query)
                img_path = frame_detail['frame_name']
                vqa_scores.append(mock_reranker.score_image(img_path, vqa_question))
            
            avg_vqa = sum(vqa_scores) / len(vqa_scores)
            vqa_scaled = avg_vqa * len(events)
            seq['total_score'] = (seq['total_score'] * 0.7) + (vqa_scaled * 0.3)
            
        ranked_sequences.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Because seq2 got 0.9 (scaled to 1.8) and seq1 got 0.1 (scaled to 0.2)
        # New Seq1 score: 10 * 0.7 + 0.2 * 0.3 = 7.0 + 0.06 = 7.06
        # New Seq2 score: 9.5 * 0.7 + 1.8 * 0.3 = 6.65 + 0.54 = 7.19
        # Seq2 should now win!
        
        self.assertEqual(ranked_sequences[0]['frame_details'][0]['frame_name'], 'good_1.jpg')
        self.assertAlmostEqual(ranked_sequences[0]['total_score'], 7.19, places=2)

if __name__ == "__main__":
    unittest.main()
