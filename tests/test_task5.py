import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, backend_dir)
sys.path.insert(0, src_dir)

sys.modules['faiss'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['open_clip'] = MagicMock()
sys.modules['open_clip'].create_model_and_transforms.return_value = (MagicMock(), MagicMock(), MagicMock())
sys.modules['PIL'] = MagicMock()

class Task5EndpointSyncTests(unittest.TestCase):
    def setUp(self):
        # Force reload modules
        modules_to_reload = [
            'src.services.user_service',
            'src.controllers.user_controller',
            'src',
            'utils.faiss_processing',
            'utils.vlm_processing',
            'utils.trake_processing'
        ]
        for m in modules_to_reload:
            if m in sys.modules:
                del sys.modules[m]

    @patch('src.controllers.user_controller.GetImageDataTrakeSearch')
    def test_temporalsearch_endpoint(self, mock_trake_search):
        """Verify that /temporalsearch route handles requests and calls temporal search."""
        mock_trake_search.return_value = [{"id": 1, "frame_key": "test_frame"}]
        
        from src import app
        client = app.test_client()
        client.testing = True
        
        # Valid temporal search payload
        payload = {
            "query": [{"query": "person walking"}],
            "topk": 10
        }
        
        res = client.post('/users/temporalsearch', json=payload)
        
        # 1. Assert status code 200 OK
        self.assertEqual(res.status_code, 200)
        
        # 2. Assert standard JSON response
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['data']['items']), 1)
        self.assertEqual(data['data']['items'][0]['frame_key'], "test_frame")
        
        # 3. Verify the underlying service was called correctly
        mock_trake_search.assert_called_once_with([{"query": "person walking"}], top_results=10)

if __name__ == '__main__':
    unittest.main()
