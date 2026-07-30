"""Unit tests for Phase 4 Task 2: Query Planner."""

import unittest
import sys
import os

# Ensure backend root is on sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.utils.nlp_processing import QueryPlanner

class TestQueryPlanner(unittest.TestCase):
    
    def test_pure_visual_query(self):
        plan = QueryPlanner.parse_query("Người đàn ông đi xe đạp màu đỏ")
        self.assertEqual(plan["visual_query"], "Người đàn ông đi xe đạp màu đỏ")
        self.assertEqual(plan["ocr_query"], "")
        self.assertEqual(plan["asr_query"], "")
        self.assertEqual(plan["weights"]["visual"], 1.0)
        self.assertEqual(plan["weights"]["ocr"], 0.0)
        
    def test_ocr_query_with_quotes(self):
        plan = QueryPlanner.parse_query('Xe cứu thương có chữ "Ambulance" chạy trên đường')
        self.assertEqual(plan["visual_query"], "Xe cứu thương có chữ  chạy trên đường")
        self.assertEqual(plan["ocr_query"], "Ambulance")
        self.assertEqual(plan["weights"]["visual"], 0.5)
        self.assertEqual(plan["weights"]["ocr"], 0.5)

    def test_asr_query(self):
        plan = QueryPlanner.parse_query("nghe tiếng còi cảnh sát hú")
        self.assertEqual(plan["asr_query"], "nghe tiếng còi cảnh sát hú")
        self.assertEqual(plan["weights"]["visual"], 0.7)
        self.assertEqual(plan["weights"]["asr"], 0.3)

    def test_multimodal_query(self):
        plan = QueryPlanner.parse_query('người đàn ông nói rằng "Cảm ơn"')
        self.assertEqual(plan["visual_query"], "người đàn ông nói rằng")
        self.assertEqual(plan["ocr_query"], "Cảm ơn")
        self.assertEqual(plan["asr_query"], "người đàn ông nói rằng")
        
        self.assertEqual(plan["weights"]["visual"], 0.4)
        self.assertEqual(plan["weights"]["ocr"], 0.3)
        self.assertEqual(plan["weights"]["asr"], 0.3)

    def test_only_ocr_query(self):
        plan = QueryPlanner.parse_query('"SALE 50%"')
        self.assertEqual(plan["visual_query"], "")
        self.assertEqual(plan["ocr_query"], "SALE 50%")
        self.assertEqual(plan["weights"]["visual"], 0.0)
        self.assertEqual(plan["weights"]["ocr"], 1.0)

if __name__ == "__main__":
    unittest.main()
