"""Unit tests for Phase 4 Task 1: Fusion Service."""

import unittest
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.services.fusion_service import normalize_scores, merge_and_rank

class TestFusionService(unittest.TestCase):
    
    def test_normalize_scores(self):
        items = [
            {"_score": 10.0},
            {"_score": 5.0},
            {"_score": 0.0}
        ]
        
        normalized = normalize_scores(items)
        
        self.assertAlmostEqual(normalized[0]["normalized_score"], 1.0)
        self.assertAlmostEqual(normalized[1]["normalized_score"], 0.5)
        self.assertAlmostEqual(normalized[2]["normalized_score"], 0.0)
        
    def test_normalize_identical_scores(self):
        # Should not divide by zero
        items = [{"_score": 5.0}, {"_score": 5.0}]
        normalized = normalize_scores(items)
        self.assertAlmostEqual(normalized[0]["normalized_score"], 0.0)

    def test_merge_and_rank(self):
        visual = [
            {"faiss_id": 1, "video_id": "V001", "normalized_score": 1.0},
            {"faiss_id": 2, "video_id": "V001", "normalized_score": 0.5},
        ]
        
        ocr = [
            {"faiss_id": 2, "ocr_text": "Hospital", "normalized_score": 1.0},
            {"faiss_id": 3, "ocr_text": "Clinic", "normalized_score": 0.8},
        ]
        
        asr = [
            {"nearest_faiss_id": 1, "text": "Ambulance", "normalized_score": 0.5},
            {"nearest_faiss_id": 3, "text": "Doctor", "normalized_score": 1.0},
        ]
        
        weights = {"visual": 0.5, "ocr": 0.3, "asr": 0.2}
        
        results = merge_and_rank(visual, ocr, asr, weights)
        
        # We expect 3 distinct items
        self.assertEqual(len(results), 3)
        
        # fid 1: V(1.0*0.5) + O(0.0) + A(0.5*0.2) = 0.5 + 0.1 = 0.6
        # fid 2: V(0.5*0.5) + O(1.0*0.3) + A(0.0) = 0.25 + 0.3 = 0.55
        # fid 3: V(0.0) + O(0.8*0.3) + A(1.0*0.2) = 0.24 + 0.2 = 0.44
        
        self.assertEqual(results[0]["faiss_id"], 1)
        self.assertAlmostEqual(results[0]["final_score"], 0.6)
        
        self.assertEqual(results[1]["faiss_id"], 2)
        self.assertAlmostEqual(results[1]["final_score"], 0.55)
        
        self.assertEqual(results[2]["faiss_id"], 3)
        self.assertAlmostEqual(results[2]["final_score"], 0.44)

if __name__ == "__main__":
    unittest.main()
