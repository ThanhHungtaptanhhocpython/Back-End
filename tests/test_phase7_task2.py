"""Unit tests for Phase 7 Task 2: Reciprocal Rank Fusion (RRF)."""

import unittest
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.services.fusion_service import reciprocal_rank_fusion

class TestReciprocalRankFusion(unittest.TestCase):
    
    def test_rrf_scoring(self):
        # Result A appears at rank 5 consistently
        # Result B appears at rank 1 in only one list, and doesn't appear in others
        
        list1 = [
            {"faiss_id": 200, "name": "B"}, # Rank 1
            {"faiss_id": 999, "name": "Trash"}, # Rank 2
            {"faiss_id": 998, "name": "Trash"}, # Rank 3
            {"faiss_id": 997, "name": "Trash"}, # Rank 4
            {"faiss_id": 100, "name": "A"}  # Rank 5
        ]
        
        list2 = [
            {"faiss_id": 996, "name": "Trash"}, # Rank 1
            {"faiss_id": 995, "name": "Trash"}, # Rank 2
            {"faiss_id": 994, "name": "Trash"}, # Rank 3
            {"faiss_id": 993, "name": "Trash"}, # Rank 4
            {"faiss_id": 100, "name": "A"}  # Rank 5
        ]
        
        list3 = [
            {"faiss_id": 992, "name": "Trash"}, # Rank 1
            {"faiss_id": 991, "name": "Trash"}, # Rank 2
            {"faiss_id": 990, "name": "Trash"}, # Rank 3
            {"faiss_id": 989, "name": "Trash"}, # Rank 4
            {"faiss_id": 100, "name": "A"}  # Rank 5
        ]
        
        lists = [list1, list2, list3]
        
        merged = reciprocal_rank_fusion(lists, k=60)
        
        # Calculate expected scores
        # A: 1/(60+5) + 1/(60+5) + 1/(60+5) = 3/65 ≈ 0.04615
        # B: 1/(60+1) = 1/61 ≈ 0.01639
        
        # B should be lower than A, even though B was rank 1 in list1!
        
        top_result = merged[0]
        self.assertEqual(top_result["faiss_id"], 100)
        self.assertEqual(top_result["name"], "A")
        
        # Verify mathematically
        self.assertAlmostEqual(top_result["rrf_score"], 3.0 / 65.0, places=5)
        
        b_result = next(item for item in merged if item["faiss_id"] == 200)
        self.assertAlmostEqual(b_result["rrf_score"], 1.0 / 61.0, places=5)

if __name__ == "__main__":
    unittest.main()
