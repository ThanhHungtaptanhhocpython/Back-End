import unittest
from unittest.mock import patch, mock_open, MagicMock
import sys
import os
import numpy as np

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

class Task4ErrorHandlingTests(unittest.TestCase):
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

    def test_faiss_missing_index_fallback(self):
        """Verify that MyFaiss doesn't crash on init if files are missing"""
        from utils.faiss_processing import MyFaiss
        import sys
        sys.modules['faiss'].read_index.side_effect = Exception("Mocked FileNotFoundError")
        
        # Test loading missing bin file
        faiss_instance = MyFaiss("dummy_missing.bin", "dummy_missing.json")
        self.assertIsNone(faiss_instance.index_clip)
        self.assertEqual(faiss_instance.id2img_fps, {})
        
        # Test search with None index doesn't crash but returns empty arrays
        scores, ids = faiss_instance._search_faiss_index(np.array([[1.0]]), k=5, index_subset=None)
        self.assertEqual(scores.size, 0)
        self.assertEqual(ids.size, 0)

    @patch('builtins.open')
    @patch('utils.faiss_processing.MyFaiss')
    @patch('utils.vlm_processing.VLMProcessor')
    def test_qnasearch_skips_missing_images(self, MockVLM, MockFaiss, mock_open_func):
        """Verify that /qnasearch gracefully skips missing image files without crashing."""
        import src.services.user_service as user_service
        
        # Setup mock behavior for search results (3 items)
        mock_infos = [
            {"global_frame_id": 1, "video_id": "V01", "split": "videos-l21-a"},
            {"global_frame_id": 2, "video_id": "V02", "split": "videos-l21-a"},
            {"global_frame_id": 3, "video_id": "V03", "split": "videos-l21-a"}
        ]
        mock_paths = ["path1.webp", "path2.webp", "path3.webp"]
        
        user_service.CosineFaiss.text_search.return_value = (None, None, mock_infos, mock_paths)
        user_service.VlmProcessorInstance.batch_answer.return_value = ["ans1", "ans3"]
        
        # Simulate open() raising an exception for the SECOND file, but succeeding for 1st and 3rd
        def open_side_effect(path, mode):
            if "path2.webp" in path:
                raise Exception("Simulated File Error")
            return mock_open(read_data=b"dummy_image_data")()
            
        mock_open_func.side_effect = open_side_effect
        
        from src import app
        client = app.test_client()
        client.testing = True
        
        res = client.post('/users/qnasearch', json={"query": "test query", "topk": 3})
        self.assertEqual(res.status_code, 200)
        
        data = res.get_json()
        self.assertTrue(data['success'])
        
        # Only 2 results should remain because the 2nd was skipped
        self.assertEqual(len(data['data']), 2)
        
        # Verify VLM was only called with the TWO successful paths
        called_paths = user_service.VlmProcessorInstance.batch_answer.call_args[0][0]
        self.assertEqual(len(called_paths), 2)
        self.assertNotIn("path2.webp", str(called_paths))

if __name__ == '__main__':
    unittest.main()
