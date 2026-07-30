import unittest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fully mock the user_service module before anything imports it
mock_user_service = MagicMock()
mock_user_service.getImageDataSingleTextSearch.return_value = []
mock_user_service.getImageDataQAndASearch.return_value = []
mock_user_service.getImageSearchById.return_value = []
mock_user_service.GetImageDataTrakeSearch.return_value = []
mock_user_service.getImageSearchByFile.return_value = []

sys.modules['src.services.user_service'] = mock_user_service

# Now import the app safely
from src import app

class Task2ValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    # --- /singletextsearch ---
    def test_singletextsearch_missing_body(self):
        res = self.client.post('/users/singletextsearch')
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Request body must be a valid JSON object", res.data)

    def test_singletextsearch_missing_query(self):
        res = self.client.post('/users/singletextsearch', json={})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Missing required field: query", res.data)

    def test_singletextsearch_invalid_topk(self):
        res = self.client.post('/users/singletextsearch', json={"query": "test", "topk": "abc"})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"topk must be a positive integer", res.data)

    def test_singletextsearch_success_default_topk(self):
        res = self.client.post('/users/singletextsearch', json={"query": "test"})
        self.assertEqual(res.status_code, 200)

    # --- /trakesearch ---
    def test_trakesearch_missing_body(self):
        res = self.client.post('/users/trakesearch')
        self.assertEqual(res.status_code, 400)

    def test_trakesearch_empty_query(self):
        res = self.client.post('/users/trakesearch', json={"query": []})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Missing required field: query", res.data)

    def test_trakesearch_invalid_event(self):
        res = self.client.post('/users/trakesearch', json={"query": [{"query": ""}]})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Each event in query list must have a non-empty", res.data)

    def test_trakesearch_success(self):
        res = self.client.post('/users/trakesearch', json={"query": [{"query": "test"}]})
        self.assertEqual(res.status_code, 200)

    # --- /imagesearch ---
    def test_imagesearch_missing_both(self):
        res = self.client.post('/users/imagesearch', data={})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Either an uploaded image file or a valid faiss_index must be provided", res.data)

    def test_imagesearch_success_faiss_id(self):
        res = self.client.post('/users/imagesearch', data={"faiss_index": "123"})
        self.assertEqual(res.status_code, 200)

if __name__ == '__main__':
    unittest.main()
