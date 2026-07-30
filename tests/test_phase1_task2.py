"""Unit tests for Phase 1 Task 2: FastAPI vs Flask Endpoint Parity.

Verifies that FastAPI and Flask respond identically to identical requests
for all search endpoints.

Run with:
    python -m pytest tests/test_phase1_task2.py -v
"""

from __future__ import annotations

import io
import json
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

# Mock faiss at the C level
_mock_faiss = MagicMock()
_mock_faiss.read_index.return_value = MagicMock(ntotal=100)
_mock_faiss.IndexIDMap2.return_value = MagicMock()
_mock_faiss.IndexFlatIP.return_value = MagicMock()
sys.modules.setdefault("faiss", _mock_faiss)

# Mock open_clip
_mock_open_clip = MagicMock()
_mock_model = MagicMock()
_mock_preprocess = MagicMock()
_mock_open_clip.create_model_and_transforms.return_value = (
    _mock_model,
    None,
    _mock_preprocess,
)
sys.modules.setdefault("open_clip", _mock_open_clip)

sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

# ---------------------------------------------------------------------------
# Helper to generate fake search results for the service mocks
# ---------------------------------------------------------------------------
def _make_mock_result(count: int = 3) -> list[dict]:
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
# Import the apps
# ---------------------------------------------------------------------------
from src import app as flask_app  # Flask app
from main import app as fastapi_app  # FastAPI app

# Patch service functions in both the service module (for FastAPI lazy imports)
# and the controller module (for Flask global imports)
_patches = [
    # Flask patches
    patch("src.controllers.user_controller.getImageDataSingleTextSearch", return_value=_make_mock_result(2)),
    patch("src.controllers.user_controller.getImageDataQAndASearch", return_value=_make_mock_result(3)),
    patch("src.controllers.user_controller.getImageSearchByFile", return_value=_make_mock_result(4)),
    patch("src.controllers.user_controller.getImageSearchById", return_value=_make_mock_result(5)),
    patch("src.controllers.user_controller.GetImageDataTrakeSearch", return_value=_make_mock_result(6)),
    
    # FastAPI patches
    patch("src.services.user_service.getImageDataSingleTextSearch", return_value=_make_mock_result(2)),
    patch("src.services.user_service.getImageDataQAndASearch", return_value=_make_mock_result(3)),
    patch("src.services.user_service.getImageSearchByFile", return_value=_make_mock_result(4)),
    patch("src.services.user_service.getImageSearchById", return_value=_make_mock_result(5)),
    patch("src.services.user_service.GetImageDataTrakeSearch", return_value=_make_mock_result(6)),
]

class TestEndpointParity(unittest.TestCase):
    """Compare Flask and FastAPI responses for all endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        flask_app.config["TESTING"] = True
        cls.flask_client = flask_app.test_client()
        # Do not raise server exceptions in TestClient to allow 400s/500s to propagate as responses
        cls.fastapi_client = TestClient(fastapi_app, raise_server_exceptions=False)
        for p in _patches:
            p.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for p in _patches:
            p.stop()

    def assertResponsesEqual(self, flask_res, fastapi_res):
        """Helper to assert that both frameworks return the same status and JSON schema."""
        self.assertEqual(
            flask_res.status_code, fastapi_res.status_code,
            f"Status code mismatch! Flask: {flask_res.status_code}, FastAPI: {fastapi_res.status_code}"
        )
        
        flask_data = json.loads(flask_res.data)
        fastapi_data = fastapi_res.json()
        
        self.assertEqual(
            flask_data.get("success"), fastapi_data.get("success"),
            "Success boolean mismatch"
        )
        
        # Check if error messages exist (they don't have to be letter-for-letter identical,
        # but both should either have it or not, and FastAPI includes Pydantic details)
        if "message" in flask_data:
            self.assertIn("message", fastapi_data)

        # In Flask, error responses set "data": None.
        # In FastAPI, our schema forces "data": {"items": [], "total_items": 0} even on errors.
        flask_data_obj = flask_data.get("data")
        fastapi_data_obj = fastapi_data.get("data")

        if flask_data_obj is None:
            # If Flask returned None for data, ensure FastAPI returned an empty data object
            self.assertIsNotNone(fastapi_data_obj)
            self.assertEqual(fastapi_data_obj.get("items", []), [])
            self.assertEqual(fastapi_data_obj.get("total_items"), 0)
        else:
            # Both should have items and total_items matching
            flask_items = flask_data_obj.get("items", [])
            fastapi_items = fastapi_data_obj.get("items", [])
            self.assertEqual(len(flask_items), len(fastapi_items))
            
            self.assertEqual(
                flask_data_obj.get("total_items"), 
                fastapi_data_obj.get("total_items")
            )

    # ------------------------------------------------------------------
    # 1. /singletextsearch
    # ------------------------------------------------------------------
    def test_singletextsearch_parity(self) -> None:
        payload = {"query": "bicycle", "topk": 10}
        
        f_res = self.flask_client.post("/users/singletextsearch", json=payload)
        fa_res = self.fastapi_client.post("/users/singletextsearch", json=payload)
        
        self.assertResponsesEqual(f_res, fa_res)

    def test_singletextsearch_invalid_parity(self) -> None:
        payload = {"query": "", "topk": -5}
        
        f_res = self.flask_client.post("/users/singletextsearch", json=payload)
        fa_res = self.fastapi_client.post("/users/singletextsearch", json=payload)
        
        self.assertResponsesEqual(f_res, fa_res)

    # ------------------------------------------------------------------
    # 2. /qnasearch
    # ------------------------------------------------------------------
    def test_qnasearch_parity(self) -> None:
        payload = {"query": "bicycle", "topk": 10}
        
        f_res = self.flask_client.post("/users/qnasearch", json=payload)
        fa_res = self.fastapi_client.post("/users/qnasearch", json=payload)
        
        self.assertResponsesEqual(f_res, fa_res)

    # ------------------------------------------------------------------
    # 3. /imagesearch
    # ------------------------------------------------------------------
    def test_imagesearch_file_parity(self) -> None:
        # Flask needs content_type="multipart/form-data" natively via data={}
        # FastAPI TestClient uses files={} for multipart
        fake_image = (io.BytesIO(b"fake_image_bytes"), "test.jpg")
        
        f_res = self.flask_client.post(
            "/users/imagesearch",
            data={"image": fake_image, "topk": "5"},
            content_type="multipart/form-data"
        )
        
        # Reset BytesIO for second request
        fake_image = ("test.jpg", io.BytesIO(b"fake_image_bytes"), "image/jpeg")
        fa_res = self.fastapi_client.post(
            "/users/imagesearch",
            data={"topk": "5"},
            files={"image": fake_image}
        )
        
        self.assertResponsesEqual(f_res, fa_res)

    def test_imagesearch_id_parity(self) -> None:
        f_res = self.flask_client.post(
            "/users/imagesearch",
            data={"faiss_index": "12345", "topk": "5"},
            content_type="multipart/form-data"
        )
        fa_res = self.fastapi_client.post(
            "/users/imagesearch",
            data={"faiss_index": "12345", "topk": "5"}
        )
        self.assertResponsesEqual(f_res, fa_res)

    def test_imagesearch_invalid_parity(self) -> None:
        # No file, no ID -> Should be 400
        f_res = self.flask_client.post(
            "/users/imagesearch",
            data={"topk": "5"},
            content_type="multipart/form-data"
        )
        fa_res = self.fastapi_client.post(
            "/users/imagesearch",
            data={"topk": "5"}
        )
        self.assertResponsesEqual(f_res, fa_res)

    # ------------------------------------------------------------------
    # 4. /temporalsearch
    # ------------------------------------------------------------------
    def test_temporalsearch_parity(self) -> None:
        payload = {
            "query": [
                {"query": "man walking"},
                {"query": "man sitting"}
            ],
            "topk": 10
        }
        f_res = self.flask_client.post("/users/temporalsearch", json=payload)
        fa_res = self.fastapi_client.post("/users/temporalsearch", json=payload)
        
        self.assertResponsesEqual(f_res, fa_res)

    # ------------------------------------------------------------------
    # 5. /ocrandodsearch
    # ------------------------------------------------------------------
    def test_ocrandodsearch_parity(self) -> None:
        f_res = self.flask_client.post("/users/ocrandodsearch")
        fa_res = self.fastapi_client.post("/users/ocrandodsearch")
        
        self.assertResponsesEqual(f_res, fa_res)


if __name__ == "__main__":
    unittest.main()
