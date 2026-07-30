"""Unit tests for Phase 1 Task 4: Clean Up Dual-Mode Architecture with Lazy Initialization.

Tests that heavy ML models are NOT loaded when the user_service module
is imported, but only when their respective getter functions are called.

Run with:
    python -m pytest tests/test_phase1_task4.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ensure backend root is on sys.path
# ---------------------------------------------------------------------------
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# ---------------------------------------------------------------------------
# Pre-mock heavy ML modules and local util classes
# ---------------------------------------------------------------------------
sys.modules.setdefault("faiss", MagicMock())
sys.modules.setdefault("open_clip", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

# Mock the actual classes inside the utils modules so we can track initialization
mock_my_faiss = MagicMock()
mock_trake = MagicMock()
mock_vlm_processor = MagicMock()

# Inject the mocked classes into sys.modules
mock_faiss_proc = MagicMock()
mock_faiss_proc.MyFaiss = mock_my_faiss
sys.modules["utils.faiss_processing"] = mock_faiss_proc

mock_trake_proc = MagicMock()
mock_trake_proc.TRAKE = mock_trake
sys.modules["utils.trake_processing"] = mock_trake_proc

mock_vlm_proc = MagicMock()
mock_vlm_proc.VLMProcessor = mock_vlm_processor
sys.modules["utils.vlm_processing"] = mock_vlm_proc


class TestLazyInitialization(unittest.TestCase):
    """Verify that models are initialized lazily, not at import time."""

    def test_lazy_loading(self) -> None:
        """Test that importing user_service doesn't trigger model init."""
        
        # 1. Reset call counts
        mock_my_faiss.reset_mock()
        mock_trake.reset_mock()
        mock_vlm_processor.reset_mock()
        
        # 2. Import the service
        from src.services import user_service
        
        # Reset any cached singletons for the test
        user_service._cosine_faiss = None
        user_service._trake_search = None
        user_service._vlm_processor = None
        
        # Importing should NOT trigger instantiations
        mock_my_faiss.assert_not_called()
        mock_trake.assert_not_called()
        mock_vlm_processor.assert_not_called()

        # 3. Call get_cosine_faiss() -> should initialize MyFaiss once
        faiss_instance_1 = user_service.get_cosine_faiss()
        mock_my_faiss.assert_called_once()
        self.assertIsNotNone(faiss_instance_1)
        
        # Calling again should NOT initialize a new instance
        faiss_instance_2 = user_service.get_cosine_faiss()
        self.assertIs(faiss_instance_1, faiss_instance_2)
        self.assertEqual(mock_my_faiss.call_count, 1)

        # 4. Call get_trake_search() -> should initialize TRAKE once
        trake_instance_1 = user_service.get_trake_search()
        mock_trake.assert_called_once_with(faiss_instance_1)
        
        trake_instance_2 = user_service.get_trake_search()
        self.assertIs(trake_instance_1, trake_instance_2)
        self.assertEqual(mock_trake.call_count, 1)

        # 5. Call get_vlm_processor() -> should initialize VLMProcessor once
        vlm_instance_1 = user_service.get_vlm_processor()
        mock_vlm_processor.assert_called_once()
        
        vlm_instance_2 = user_service.get_vlm_processor()
        self.assertIs(vlm_instance_1, vlm_instance_2)
        self.assertEqual(mock_vlm_processor.call_count, 1)


if __name__ == "__main__":
    unittest.main()
