import unittest
from flask import Flask
from src.controllers.user_controller import users
import json

app = Flask(__name__)
app.register_blueprint(users)

class TemporalValidationTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_missing_json(self):
        resp = self.client.post('/trakesearch', data="not json", content_type='text/plain')
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["message"], "Request must be JSON.")

    def test_missing_query(self):
        resp = self.client.post('/trakesearch', json={})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["message"], "Query must be a non-empty list of events.")

    def test_invalid_query_shape(self):
        resp = self.client.post('/trakesearch', json={"query": ["just string"]})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["message"], "Each event in query list must have a non-empty 'query' string.")

    def test_invalid_topk(self):
        resp = self.client.post('/trakesearch', json={"query": [{"query": "event"}], "topk": "abc"})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["message"], "topk must be a positive integer.")

    def test_invalid_topk_negative(self):
        resp = self.client.post('/trakesearch', json={"query": [{"query": "event"}], "topk": -5})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertEqual(data["message"], "topk must be a positive integer.")

if __name__ == '__main__':
    unittest.main()
