"""Unit tests for Task 1: Image Search API endpoint logic.

These tests mock heavy dependencies (Faiss C library, OpenCLIP, torch)
at the sys.modules level BEFORE any application code is imported, so the
Flask app can start without requiring index files or GPU models.

Run with:
    python -m pytest tests/test_task1_imagesearch.py -v
"""

from __future__ import annotations

import io
import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Ensure backend root is on sys.path
# ---------------------------------------------------------------------------
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# ---------------------------------------------------------------------------
# 2. Pre-patch heavy C-extension modules BEFORE any app code is imported.
#    This prevents faiss.read_index, open_clip.create_model_and_transforms,
#    and torch from executing real GPU/disk operations.
# ---------------------------------------------------------------------------

# Mock faiss at the C level
_mock_faiss = MagicMock()
_mock_faiss.read_index.return_value = MagicMock(ntotal=100)
_mock_faiss.IndexIDMap2.return_value = MagicMock()
_mock_faiss.IndexFlatIP.return_value = MagicMock()
sys.modules.setdefault("faiss", _mock_faiss)

# Mock open_clip
_mock_open_clip = MagicMock()
_mock_model = MagicMock()
_mock_model.encode_text.return_value = MagicMock(
    norm=MagicMock(return_value=MagicMock()),
    cpu=MagicMock(return_value=MagicMock()),
)
_mock_preprocess = MagicMock()
_mock_open_clip.create_model_and_transforms.return_value = (
    _mock_model,
    None,
    _mock_preprocess,
)
_mock_open_clip.get_tokenizer.return_value = MagicMock()
sys.modules.setdefault("open_clip", _mock_open_clip)

# Mock torch (minimal stubs)
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.no_grad.return_value.__enter__ = MagicMock()
_mock_torch.no_grad.return_value.__exit__ = MagicMock()
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())

# Mock transformers (for VLMProcessor)
sys.modules.setdefault("transformers", MagicMock())

# Mock googletrans (for Translation)
_mock_googletrans = MagicMock()
sys.modules.setdefault("googletrans", _mock_googletrans)

# Mock sentence_transformers
sys.modules.setdefault("sentence_transformers", MagicMock())


# ---------------------------------------------------------------------------
# 3. Helper to generate fake search results
# ---------------------------------------------------------------------------
def _make_mock_result(count: int = 3) -> list[dict]:
    """Generate a fake result list that mimics service return values."""
    return [
        {
            "id": i,
            "folder_key": "L26",
            "video_key": f"V{i:03d}",
            "frame_key": 10000 + i,
            "image": "ZmFrZV9iYXNlNjQ=",
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# 4. Now it is safe to import the Flask app and patch service functions
# ---------------------------------------------------------------------------
# Import will trigger user_service.py module-level init, but faiss.read_index
# is already mocked so it won't crash.
from src import app as flask_app  # noqa: E402

# Patch the service-level functions that the controller calls
_file_search_patcher = patch(
    "src.controllers.user_controller.getImageSearchByFile",
    return_value=_make_mock_result(3),
)
_id_search_patcher = patch(
    "src.controllers.user_controller.getImageSearchById",
    return_value=_make_mock_result(2),
)
mock_file_search = _file_search_patcher.start()
mock_id_search = _id_search_patcher.start()


class TestImageSearchEndpoint(unittest.TestCase):
    """Tests for POST /users/imagesearch controller logic."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a Flask test client."""
        flask_app.config["TESTING"] = True
        cls.client = flask_app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        _file_search_patcher.stop()
        _id_search_patcher.stop()

    # ------------------------------------------------------------------
    # Test 1: Upload an image file -> should call getImageSearchByFile
    # ------------------------------------------------------------------
    def test_image_upload_returns_200(self) -> None:
        """Uploading an image file should trigger file-based search."""
        fake_image = (io.BytesIO(b"fake_image_bytes"), "test.jpg")
        response = self.client.post(
            "/users/imagesearch",
            data={"image": fake_image, "topk": "5"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.data)
        self.assertTrue(body["success"])
        self.assertIn("items", body["data"])
        self.assertEqual(body["data"]["total_items"], 3)

    # ------------------------------------------------------------------
    # Test 2: Provide faiss_index (no file) -> should call getImageSearchById
    # ------------------------------------------------------------------
    def test_faiss_index_returns_200(self) -> None:
        """Providing a faiss_index without a file should use ID-based search."""
        response = self.client.post(
            "/users/imagesearch",
            data={"faiss_index": "36244", "topk": "5"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.data)
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["total_items"], 2)

    # ------------------------------------------------------------------
    # Test 3: No file AND no faiss_index -> 400
    # ------------------------------------------------------------------
    def test_no_input_returns_400(self) -> None:
        """Sending neither image nor faiss_index should return 400."""
        response = self.client.post(
            "/users/imagesearch",
            data={"topk": "5"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.data)
        self.assertFalse(body["success"])
        self.assertIn("Either an uploaded image file", body["message"])

    # ------------------------------------------------------------------
    # Test 4: Invalid topk (negative) -> 400
    # ------------------------------------------------------------------
    def test_negative_topk_returns_400(self) -> None:
        """A negative topk value should return 400 Bad Request."""
        fake_image = (io.BytesIO(b"fake_image_bytes"), "test.jpg")
        response = self.client.post(
            "/users/imagesearch",
            data={"image": fake_image, "topk": "-1"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.data)
        self.assertFalse(body["success"])
        self.assertIn("topk must be a positive integer", body["message"])

    # ------------------------------------------------------------------
    # Test 5: Invalid topk (string) -> 400
    # ------------------------------------------------------------------
    def test_non_numeric_topk_returns_400(self) -> None:
        """A non-numeric topk value should return 400 Bad Request."""
        fake_image = (io.BytesIO(b"fake_image_bytes"), "test.jpg")
        response = self.client.post(
            "/users/imagesearch",
            data={"image": fake_image, "topk": "abc"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.data)
        self.assertFalse(body["success"])

    # ------------------------------------------------------------------
    # Test 6: Invalid faiss_index (non-integer string) -> 400
    # ------------------------------------------------------------------
    def test_non_integer_faiss_index_returns_400(self) -> None:
        """A non-integer faiss_index should return 400 Bad Request."""
        response = self.client.post(
            "/users/imagesearch",
            data={"faiss_index": "not_a_number", "topk": "5"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.data)
        self.assertFalse(body["success"])
        self.assertIn("faiss_index must be an integer", body["message"])

    # ------------------------------------------------------------------
    # Test 7: Default topk when omitted -> should default to 100
    # ------------------------------------------------------------------
    def test_default_topk(self) -> None:
        """Omitting topk should default to 100 and succeed."""
        fake_image = (io.BytesIO(b"fake_image_bytes"), "test.jpg")
        response = self.client.post(
            "/users/imagesearch",
            data={"image": fake_image},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)

    # ------------------------------------------------------------------
    # Test 8: Response schema validation
    # ------------------------------------------------------------------
    def test_response_schema(self) -> None:
        """Response body must follow the unified schema from AGENTS.md."""
        fake_image = (io.BytesIO(b"fake_image_bytes"), "test.jpg")
        response = self.client.post(
            "/users/imagesearch",
            data={"image": fake_image, "topk": "3"},
            content_type="multipart/form-data",
        )
        body = json.loads(response.data)

        # Top-level keys
        self.assertIn("success", body)
        self.assertIn("data", body)

        # Data structure
        self.assertIn("items", body["data"])
        self.assertIn("total_items", body["data"])
        self.assertIsInstance(body["data"]["items"], list)
        self.assertIsInstance(body["data"]["total_items"], int)


if __name__ == "__main__":
    unittest.main()
