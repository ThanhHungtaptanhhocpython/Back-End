import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import types
import os
import numpy as np

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, backend_dir)
sys.path.insert(0, src_dir)

# Mock heavy deps only if they are not already imported for real elsewhere
# (clobbering real torch/faiss breaks later tests that need them).
sys.modules.setdefault('faiss', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('transformers', MagicMock())
_open_clip = sys.modules.setdefault('open_clip', MagicMock())
if isinstance(_open_clip, MagicMock):
    _open_clip.create_model_and_transforms.return_value = (MagicMock(), MagicMock(), MagicMock())
sys.modules.setdefault('PIL', MagicMock())

# test_task2.py stubs sys.modules['src.services.user_service'] with a MagicMock;
# drop that stub so requests below hit the real service implementation.
_stub = sys.modules.get("src.services.user_service")
if _stub is not None and not isinstance(_stub, types.ModuleType):
    del sys.modules["src.services.user_service"]

class Task4ErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        # Drop cached modules so each test starts from a clean import state
        modules_to_reload = [
            'src.services.user_service',
            'utils.faiss_processing',
            'utils.vlm_processing',
            'utils.trake_processing'
        ]
        for m in modules_to_reload:
            sys.modules.pop(m, None)

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

    def _fake_beit3_module(self, retriever):
        """Return a stub module for src.services.beit3_retriever whose factory
        yields `retriever`. The real module cannot be imported here because the
        heavy deps above are mocked."""
        mod = types.ModuleType("src.services.beit3_retriever")
        mod.get_beit3_retriever = MagicMock(return_value=retriever)
        return mod

    def test_qnasearch_defaults_missing_answers(self):
        """Verify that /qnasearch fills in an empty answer for items that omit it."""
        mock_retriever = MagicMock()
        # The first item intentionally omits the optional 'answer' field.
        mock_retriever.search_visual.return_value = [
            {"faiss_id": 101, "video_key": "V01", "frame_key": "L21_V001_0001"},
            {"faiss_id": 102, "video_key": "V02", "frame_key": "L21_V002_0002", "answer": "prefilled"},
        ]
        fake_module = self._fake_beit3_module(mock_retriever)

        from main import app
        client = TestClient(app)

        with patch.dict(sys.modules, {"src.services.beit3_retriever": fake_module}):
            res = client.post('/users/qnasearch', json={"query": "test query", "topk": 2})

        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['total_items'], 2)

        # Missing answers are defaulted to '', existing ones are preserved
        self.assertEqual(data['data']['items'][0]['answer'], '')
        self.assertEqual(data['data']['items'][1]['answer'], 'prefilled')

    def test_qnasearch_returns_sanitized_error_on_failure(self):
        """Verify that a retriever failure produces a sanitized JSON error instead of a crash."""
        mock_retriever = MagicMock()
        mock_retriever.search_visual.side_effect = RuntimeError("index unavailable")
        fake_module = self._fake_beit3_module(mock_retriever)

        from main import app
        client = TestClient(app, raise_server_exceptions=False)

        with patch.dict(sys.modules, {"src.services.beit3_retriever": fake_module}):
            res = client.post('/users/qnasearch', json={"query": "test query", "topk": 1})

        self.assertEqual(res.status_code, 500)

        data = res.json()
        self.assertFalse(data['success'])
        # Raw exception details must not leak to the client
        self.assertNotIn("index unavailable", data['message'])
        self.assertEqual(data['data']['items'], [])

if __name__ == '__main__':
    unittest.main()
