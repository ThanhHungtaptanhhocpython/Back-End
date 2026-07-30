"""Unit tests for Phase 6 Task 1: Beam Search Temporal Ranking."""

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

class TestBeamSearch(unittest.TestCase):
    
    def test_beam_search_limits_width(self):
        trake = TRAKE(MockFaiss())
        
        # Create mock event candidates for 3 events
        # Event 1 has 100 candidates
        # Event 2 has 100 candidates
        # Event 3 has 100 candidates
        event1 = [{"frame_name": f"img_1_{i}", "global_frame_id": i, "score": 1.0, "split": "a"} for i in range(100)]
        event2 = [{"frame_name": f"img_2_{i}", "global_frame_id": i+200, "score": 1.0, "split": "a"} for i in range(100)]
        event3 = [{"frame_name": f"img_3_{i}", "global_frame_id": i+400, "score": 1.0, "split": "a"} for i in range(100)]
        
        event_candidates = [event1, event2, event3]
        
        # With beam_width=5, it should evaluate top 5 at each step.
        # So final sequences should be exactly 5, not 100*100*100 = 1,000,000.
        sequences = trake.beam_search_sequences("V001", event_candidates, beam_width=5)
        
        self.assertEqual(len(sequences), 5)
        # All of them should have 3 frames
        for seq in sequences:
            self.assertEqual(len(seq['frames']), 3)
            # Ensure chronological order
            self.assertTrue(seq['global_frame_ids'][0] < seq['global_frame_ids'][1] < seq['global_frame_ids'][2])

if __name__ == "__main__":
    unittest.main()
