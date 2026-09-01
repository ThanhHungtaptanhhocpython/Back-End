"""Unit tests for Phase 1 Task 1: Pydantic Settings Configuration.

Tests that the Settings class loads defaults correctly, that environment
variable overrides work, and that path resolver methods return expected values.

Heavy ML dependencies (faiss, open_clip, torch, transformers) are mocked
at sys.modules level before any app code is imported.

Run with:
    python -m pytest tests/test_phase1_task1.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure backend root is on sys.path
# ---------------------------------------------------------------------------
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# ---------------------------------------------------------------------------
# Pre-mock heavy C-extension modules to prevent import chain crashes
# ---------------------------------------------------------------------------
_mock_faiss = MagicMock()
_mock_faiss.read_index.return_value = MagicMock(ntotal=100)
sys.modules.setdefault("faiss", _mock_faiss)

_mock_open_clip = MagicMock()
_mock_open_clip.create_model_and_transforms.return_value = (MagicMock(), None, MagicMock())
_mock_open_clip.get_tokenizer.return_value = MagicMock()
sys.modules.setdefault("open_clip", _mock_open_clip)

_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())

sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("googletrans", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

# ---------------------------------------------------------------------------
# Now safe to import settings
# ---------------------------------------------------------------------------
from src.config.settings import Settings, get_settings  # noqa: E402


class TestSettingsDefaults(unittest.TestCase):
    """Verify that default values are sensible without any .env file."""

    def setUp(self) -> None:
        self.settings = Settings(_env_file=None, debug=False, faiss_index_path=None, metadata_path=None, keyframes_root=None, features_root=None)

    def test_default_env(self) -> None:
        self.assertEqual(self.settings.env, "development")

    def test_default_debug(self) -> None:
        self.assertFalse(self.settings.debug)

    def test_default_host(self) -> None:
        self.assertEqual(self.settings.host, "0.0.0.0")

    def test_default_port(self) -> None:
        self.assertEqual(self.settings.port, 3000)

    def test_default_clip_model(self) -> None:
        self.assertEqual(self.settings.clip_model_name, "ViT-H-14-quickgelu")

    def test_default_clip_pretrained(self) -> None:
        self.assertEqual(self.settings.clip_pretrained, "dfn5b")

    def test_default_log_level(self) -> None:
        self.assertEqual(self.settings.log_level, "INFO")


class TestSettingsPathResolvers(unittest.TestCase):
    """Verify that path resolver methods return correct defaults."""

    def setUp(self) -> None:
        self.settings = Settings(_env_file=None, debug=False, faiss_index_path=None, metadata_path=None, keyframes_root=None, features_root=None)

    def test_faiss_index_default_path(self) -> None:
        path = self.settings.get_faiss_index_path()
        self.assertTrue(str(path).endswith("faiss_index.bin"))
        self.assertIn("dict", str(path))

    def test_metadata_default_path(self) -> None:
        path = self.settings.get_metadata_path()
        self.assertTrue(str(path).endswith("metadata_clip.json"))
        self.assertIn("dict", str(path))

    def test_keyframes_default_path(self) -> None:
        path = self.settings.get_keyframes_root()
        self.assertTrue(str(path).endswith("Keyframes"))
        self.assertIn("data", str(path))

    def test_features_default_path(self) -> None:
        path = self.settings.get_features_root()
        self.assertTrue(str(path).endswith("features"))
        self.assertIn("data", str(path))


class TestSettingsEnvOverride(unittest.TestCase):
    """Verify that environment variables override default values."""

    @patch.dict(os.environ, {"PORT": "8080", "HOST": "127.0.0.1", "DEBUG": "true"})
    def test_override_server_settings(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.port, 8080)
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertTrue(settings.debug)

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_override_log_level(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.log_level, "DEBUG")

    @patch.dict(os.environ, {"CLIP_MODEL_NAME": "ViT-B-32", "CLIP_PRETRAINED": "openai"})
    def test_override_model_config(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(settings.clip_model_name, "ViT-B-32")
        self.assertEqual(settings.clip_pretrained, "openai")

    @patch.dict(os.environ, {"FAISS_INDEX_PATH": "/custom/path/index.bin"})
    def test_override_faiss_path(self) -> None:
        settings = Settings(_env_file=None)
        path = settings.get_faiss_index_path()
        self.assertEqual(path.as_posix(), "/custom/path/index.bin")

    @patch.dict(os.environ, {"METADATA_PATH": "/custom/metadata.json"})
    def test_override_metadata_path(self) -> None:
        settings = Settings(_env_file=None)
        path = settings.get_metadata_path()
        self.assertEqual(path.as_posix(), "/custom/metadata.json")

    @patch.dict(os.environ, {"MEDIA_INFO_PATH": "/custom/media-info"})
    def test_override_media_info_path(self) -> None:
        settings = Settings(_env_file=None)
        path = settings.get_media_info_path()
        self.assertEqual(path.as_posix(), "/custom/media-info")


class TestGetSettingsSingleton(unittest.TestCase):
    """Verify the lru_cache singleton behavior."""

    def test_returns_same_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        self.assertIs(s1, s2)

    def test_cache_clear_creates_new_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        self.assertIsNot(s1, s2)


class TestSettingsFieldTypes(unittest.TestCase):
    """Verify that fields are parsed to the correct Python types."""

    def test_port_is_int(self) -> None:
        settings = Settings(_env_file=None)
        self.assertIsInstance(settings.port, int)

    def test_debug_is_bool(self) -> None:
        settings = Settings(_env_file=None)
        self.assertIsInstance(settings.debug, bool)

    def test_src_dir_is_path(self) -> None:
        settings = Settings(_env_file=None)
        self.assertIsInstance(settings.src_dir, Path)

    @patch.dict(os.environ, {"PORT": "not_a_number"})
    def test_invalid_port_raises(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Settings(_env_file=None)


if __name__ == "__main__":
    unittest.main()
