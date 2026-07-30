import unittest
from flask import Flask
from unittest.mock import patch, MagicMock
import sys
import json

sys.modules['faiss'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['open_clip'] = MagicMock()
sys.modules['open_clip'].create_model_and_transforms.return_value = (MagicMock(), MagicMock(), MagicMock())
sys.modules['PIL'] = MagicMock()

from src.controllers.user_controller import users

app = Flask(__name__)
app.register_blueprint(users)

class TemporalValidationTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_missing_json(self):
        resp = self.client.post('/temporalsearch', data="not json", content_type='text/plain')
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["error"], "Request body must be a valid JSON object.")

    def test_missing_query(self):
        resp = self.client.post('/temporalsearch', json={})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["error"], "Missing required field: query (must be a non-empty list of events).")

    def test_invalid_query_shape(self):
        resp = self.client.post('/temporalsearch', json={"query": ["just string"]})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["error"], "Each event in query list must have a non-empty 'query' string.")

    def test_invalid_topk(self):
        resp = self.client.post('/temporalsearch', json={"query": [{"query": "event"}], "topk": "abc"})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["error"], "topk must be a positive integer.")

    def test_invalid_topk_negative(self):
        resp = self.client.post('/temporalsearch', json={"query": [{"query": "event"}], "topk": -5})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["error"], "topk must be a positive integer.")

if __name__ == '__main__':
    unittest.main()
