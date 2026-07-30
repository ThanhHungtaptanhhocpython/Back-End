"""Unit tests for Phase 6 Task 2: Exponential Decay."""

import unittest
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.trake_processing import TRAKE

class MockFaiss:
    pass

class TestExponentialDecay(unittest.TestCase):
    
    def test_decay_penalizes_long_gaps(self):
        trake = TRAKE(MockFaiss())
        
        # Candidate A and B form a sequence with 10s gap
        cand1 = {"frame_name": "img1", "global_frame_id": 10, "score": 1.0, "split": "a", "timestamp": 0.0}
        cand2_short = {"frame_name": "img2_short", "global_frame_id": 20, "score": 1.0, "split": "a", "timestamp": 10.0}
        cand2_long = {"frame_name": "img2_long", "global_frame_id": 30, "score": 1.0, "split": "a", "timestamp": 300.0}
        
        event1 = [cand1]
        event2 = [cand2_short, cand2_long]
        
        event_candidates = [event1, event2]
        
        sequences = trake.beam_search_sequences("V001", event_candidates, beam_width=5)
        
        # Both sequences have base score of 2.0
        # The short sequence (10s gap) should have higher total_score than long sequence (300s gap)
        self.assertEqual(len(sequences), 2)
        
        # Since they are sorted descending by score, the first one should be the short gap
        self.assertEqual(sequences[0]['frames'][1], "img2_short")
        self.assertEqual(sequences[1]['frames'][1], "img2_long")
        
        # Verify scores mathematically
        import math
        short_expected = 2.0 * math.exp(-0.01 * 10.0)
        long_expected = 2.0 * math.exp(-0.01 * 300.0)
        
        self.assertAlmostEqual(sequences[0]['total_score'], short_expected, places=5)
        self.assertAlmostEqual(sequences[1]['total_score'], long_expected, places=5)

if __name__ == "__main__":
    unittest.main()
