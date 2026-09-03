import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import types
import os

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, backend_dir)
sys.path.insert(0, src_dir)

# Mock heavy deps only if they are not already imported for real elsewhere
# (clobbering real torch/faiss breaks later tests that need them).
sys.modules.setdefault('faiss', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('transformers', MagicMock())
sys.modules.setdefault('PIL', MagicMock())

# test_task2.py stubs sys.modules['src.services.user_service'] with a MagicMock;
# drop that stub so this module exercises the real service implementation.
_stub = sys.modules.get("src.services.user_service")
if _stub is not None and not isinstance(_stub, types.ModuleType):
    del sys.modules["src.services.user_service"]

import src.services.user_service as user_service


def _make_fake_beit3_module(retriever):
    """Return a stub module for src.services.beit3_retriever whose factory
    yields `retriever`. The real module cannot be imported here because the
    heavy deps above are mocked."""
    mod = types.ModuleType("src.services.beit3_retriever")
    mod.get_beit3_retriever = MagicMock(return_value=retriever)
    return mod


class Task3SingletonTests(unittest.TestCase):
    def setUp(self):
        # Start each test from a clean singleton state
        self._reset_singletons()

    def _reset_singletons(self):
        user_service._trake_search = None
        user_service._vlm_processor = None
        user_service._elastic_processor = None

    @patch('src.services.user_service.TRAKE')
    @patch('src.services.user_service.VLMProcessor')
    def test_lazy_singletons_initialized_once(self, MockVLM, MockTRAKE):
        """
        Verify that the lazy service getters create each dependency exactly once
        and cache it.
        """
        trake = user_service.get_trake_search()
        MockTRAKE.assert_called_once_with()

        vlm = user_service.get_vlm_processor()
        self.assertIs(vlm, user_service.get_vlm_processor())
        MockVLM.assert_called_once()

    def test_qnasearch_reuses_retriever_instance(self):
        """
        Verify that /qnasearch serves every request through the same BEiT3
        retriever instance (the module-level singleton) on both URL prefixes.
        """
        mock_retriever = MagicMock()
        mock_retriever.search_visual.return_value = [
            {"faiss_id": 101, "video_key": "V01", "frame_key": "L21_V001_0001"}
        ]
        fake_module = _make_fake_beit3_module(mock_retriever)

        from main import app
        client = TestClient(app)

        with patch.dict(sys.modules, {"src.services.beit3_retriever": fake_module}), \
             patch("src.services.retrieval_backend.active_backend", lambda settings=None: "beit3"):
            res1 = client.post('/users/qnasearch', json={"query": "test query", "topk": 1})
            res2 = client.post('/qnasearch', json={"query": "another query", "topk": 1})

        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res1.json()["success"])
        self.assertEqual(res1.json()["data"]["total_items"], 1)
        self.assertEqual(res1.json()["data"]["items"][0]["faiss_id"], 101)

        # Both requests were served by the same retriever instance
        self.assertEqual(mock_retriever.search_visual.call_count, 2)

if __name__ == '__main__':
    unittest.main()
