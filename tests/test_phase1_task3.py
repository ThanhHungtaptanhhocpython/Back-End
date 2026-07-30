"""Unit tests for Phase 1 Task 3: Middleware and Exception Handling.

Tests that RequestLoggingMiddleware adds X-Request-ID headers and
that the global_exception_handler catches generic Exceptions and 
returns the unified BaseResponse error schema.

Run with:
    python -m pytest tests/test_phase1_task3.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Ensure backend root is on sys.path
# ---------------------------------------------------------------------------
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# ---------------------------------------------------------------------------
# Pre-mock heavy ML modules (Faiss, Torch) BEFORE importing FastAPI app
# ---------------------------------------------------------------------------
sys.modules.setdefault("faiss", MagicMock())
sys.modules.setdefault("open_clip", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

# ---------------------------------------------------------------------------
# Import the FastAPI app
# ---------------------------------------------------------------------------
from main import app  # noqa: E402


class TestMiddleware(unittest.TestCase):
    """Verify middleware behavior (Logging, X-Request-ID, Error Handling)."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_request_id_in_headers(self) -> None:
        """Verify that every successful request gets an X-Request-ID header."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-request-id", response.headers)
        self.assertTrue(len(response.headers["x-request-id"]) > 10)

    @patch("src.api.middleware.logger.info")
    def test_request_logging(self, mock_logger_info: MagicMock) -> None:
        """Verify that the middleware logs the request timing and ID."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        # Check that logger.info was called with the expected format
        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        
        self.assertIn("GET /health", log_message)
        self.assertIn("status=200", log_message)
        self.assertIn("duration=", log_message)
        self.assertIn("ms", log_message)
        self.assertIn("request_id=", log_message)
        self.assertIn(response.headers["x-request-id"], log_message)


class TestGlobalExceptionHandler(unittest.TestCase):
    """Verify that unhandled exceptions are caught and formatted properly."""

    @classmethod
    def setUpClass(cls) -> None:
        """Add a temporary route to the app for testing exceptions."""
        @app.get("/test-error")
        def _test_error_route():
            raise RuntimeError("Simulated unexpected database failure!")

    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_global_exception_handler_returns_standard_schema(self) -> None:
        """Simulate an unhandled exception in an endpoint."""
        response = self.client.get("/test-error")
        
        # It should return 500, not crash the server
        self.assertEqual(response.status_code, 500)
        
        # It should still have an X-Request-ID header (middleware runs outside exception handler)
        self.assertIn("x-request-id", response.headers)
        
        # It should match the standard BaseResponse schema
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["message"], "Internal server error. Please try again later.")
        self.assertIn("data", data)
        self.assertIn("items", data["data"])
        self.assertEqual(data["data"]["items"], [])
        self.assertEqual(data["data"]["total_items"], 0)


if __name__ == "__main__":
    unittest.main()
