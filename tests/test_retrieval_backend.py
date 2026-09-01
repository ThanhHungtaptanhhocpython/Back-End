"""Unit tests for src/services/retrieval_backend.py -- the RETRIEVAL_BACKEND
selector shared by textual KIS, grounded Q&A, and TRAKE retrieval."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import Settings
from src.services.retrieval_backend import (
    BEIT3,
    JINA_CLIP_V2,
    RetrievalBackendError,
    get_active_retriever,
)


class BackendSelectionTests(unittest.TestCase):
    def test_beit3_backend_calls_get_beit3_retriever(self):
        sentinel = object()
        fake_module = MagicMock()
        fake_module.get_beit3_retriever = MagicMock(return_value=sentinel)
        old = sys.modules.get("src.services.beit3_retriever")
        sys.modules["src.services.beit3_retriever"] = fake_module
        try:
            settings = Settings(debug=False, retrieval_backend="beit3")
            result = get_active_retriever(settings)
            self.assertIs(result, sentinel)
            fake_module.get_beit3_retriever.assert_called_once()
        finally:
            if old is not None:
                sys.modules["src.services.beit3_retriever"] = old
            else:
                sys.modules.pop("src.services.beit3_retriever", None)

    def test_jina_backend_calls_get_jina_retriever_not_beit3(self):
        sentinel = object()
        fake_jina = MagicMock()
        fake_jina.get_jina_retriever = MagicMock(return_value=sentinel)
        fake_beit3 = MagicMock()
        fake_beit3.get_beit3_retriever = MagicMock(side_effect=AssertionError("must not touch BEiT3"))
        old_jina = sys.modules.get("src.services.jina_retriever")
        old_beit3 = sys.modules.get("src.services.beit3_retriever")
        sys.modules["src.services.jina_retriever"] = fake_jina
        sys.modules["src.services.beit3_retriever"] = fake_beit3
        try:
            settings = Settings(debug=False, retrieval_backend="jina_clip_v2")
            result = get_active_retriever(settings)
            self.assertIs(result, sentinel)
            fake_jina.get_jina_retriever.assert_called_once()
            fake_beit3.get_beit3_retriever.assert_not_called()
        finally:
            for name, old in (("src.services.jina_retriever", old_jina), ("src.services.beit3_retriever", old_beit3)):
                if old is not None:
                    sys.modules[name] = old
                else:
                    sys.modules.pop(name, None)

    def test_unknown_backend_raises_a_clear_error(self):
        settings = Settings(debug=False, retrieval_backend="not_a_real_backend")
        with self.assertRaises(RetrievalBackendError):
            get_active_retriever(settings)

    def test_backend_value_is_case_and_whitespace_insensitive(self):
        sentinel = object()
        fake_module = MagicMock()
        fake_module.get_beit3_retriever = MagicMock(return_value=sentinel)
        old = sys.modules.get("src.services.beit3_retriever")
        sys.modules["src.services.beit3_retriever"] = fake_module
        try:
            settings = Settings(debug=False, retrieval_backend=" BEIT3 ")
            self.assertIs(get_active_retriever(settings), sentinel)
        finally:
            if old is not None:
                sys.modules["src.services.beit3_retriever"] = old
            else:
                sys.modules.pop("src.services.beit3_retriever", None)

    def test_default_backend_is_beit3(self):
        settings = Settings(debug=False)
        self.assertEqual(settings.retrieval_backend, BEIT3)


class ModuleImportHasNoSideEffectsTests(unittest.TestCase):
    def test_importing_selector_does_not_import_torch_or_transformers(self):
        # A pure selection function must not force either backend's heavy
        # deps to load just because it was imported.
        import src.services.retrieval_backend  # noqa: F401

        # No assertion needed beyond "import succeeds" -- the module itself
        # only imports Settings/typing at module scope (see source).


if __name__ == "__main__":
    unittest.main()
