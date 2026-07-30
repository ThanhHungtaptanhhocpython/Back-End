import unittest
from unittest.mock import patch, MagicMock
import sys
import os

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, backend_dir)
sys.path.insert(0, src_dir)

# 1. Mock heavy dependencies globally so they don't crash
sys.modules['faiss'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['open_clip'] = MagicMock()
sys.modules['PIL'] = MagicMock()

class Task3SingletonTests(unittest.TestCase):
    def setUp(self):
        # Force reload modules to ensure we capture the initialization logic
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

    @patch('utils.faiss_processing.MyFaiss')
    @patch('utils.vlm_processing.VLMProcessor')
    def test_singleton_initialization(self, MockVLM, MockFaiss):
        """
        Verify that MyFaiss and VLMProcessor are instantiated exactly once at module load,
        and that TRAKE reuses the Faiss instance.
        """
        import src.services.user_service as user_service
        
        # Verify singletons are created at module level
        self.assertIsNotNone(user_service.CosineFaiss)
        self.assertIsNotNone(user_service.TrakeSearch)
        self.assertIsNotNone(user_service.VlmProcessorInstance)
        
        # Verify TRAKE reuses CosineFaiss (Dependency Injection)
        self.assertIs(user_service.TrakeSearch.faiss_searcher, user_service.CosineFaiss)
        
        # Verify VLMProcessor was instantiated exactly ONCE during import
        MockVLM.assert_called_once()
        
        # Verify MyFaiss was instantiated exactly ONCE (for CosineFaiss)
        # If TRAKE was creating its own Faiss, this would fail.
        MockFaiss.assert_called_once()
        
    @patch('utils.faiss_processing.MyFaiss')
    @patch('utils.vlm_processing.VLMProcessor')
    def test_qnasearch_uses_singleton(self, MockVLM, MockFaiss):
        """
        Verify that calling the /qnasearch endpoint does NOT instantiate VLMProcessor again.
        """
        import src.services.user_service as user_service
        
        # Setup mock behavior for the global instance
        user_service.VlmProcessorInstance.batch_answer.return_value = ["mock answer"]
        
        # We also need to mock CosineFaiss.text_search to prevent errors in endpoint
        user_service.CosineFaiss.text_search.return_value = ([], [], [], [])
        
        from src import app
        client = app.test_client()
        client.testing = True
        
        # Reset mock call counts after module initialization
        MockVLM.reset_mock()
        
        # Make a request
        res = client.post('/users/qnasearch', json={"query": "test query", "topk": 1})
        
        self.assertEqual(res.status_code, 200)
        
        # VERY IMPORTANT: Verify VLMProcessor was NOT instantiated during the request
        MockVLM.assert_not_called()
        
        # Verify the singleton instance's batch_answer was used instead
        user_service.VlmProcessorInstance.batch_answer.assert_called_once()

if __name__ == '__main__':
    unittest.main()
