"""Pydantic Settings configuration for the FastAPI application.

This module replaces the Flask-style config classes (dev_config.py,
production.py) with a single Pydantic BaseSettings class that reads
from environment variables and .env files automatically.

Usage:
    from src.config.settings import get_settings
    settings = get_settings()
    print(settings.host, settings.port)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


def _default_src_dir() -> Path:
    """Return the absolute path to the `src/` directory."""
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        env: The current environment name (development, production).
        debug: Whether to run in debug/reload mode.
        host: The host address to bind the server to.
        port: The port number to bind the server to.
        src_dir: Absolute path to the `src/` directory. All data paths
            are resolved relative to this.
        faiss_index_path: Path to the Faiss binary index file.
        metadata_path: Path to the metadata JSON file.
        keyframes_root: Root directory for keyframe images.
        features_root: Root directory for per-video .npy feature files.
        clip_model_name: OpenCLIP model architecture name.
        clip_pretrained: OpenCLIP pretrained weights identifier.
        log_level: Python logging level string.
    """

    # --- Server ---
    env: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 3000

    # --- Data Paths (resolved relative to src_dir if not absolute) ---
    src_dir: Path = _default_src_dir()
    faiss_index_path: Path | None = None
    metadata_path: Path | None = None
    keyframes_root: Path | None = None
    features_root: Path | None = None

    # --- External Services ---
    elasticsearch_url: str = "http://localhost:9200"

    # --- Model Configuration ---
    clip_model_name: str = "ViT-H-14-quickgelu"
    clip_pretrained: str = "dfn5b"

    # --- BEiT3 Retrieval (real visual-search path) ---
    # All paths are machine-specific runtime artifacts and must be set via
    # the environment; there is no in-repo default because the checkpoint,
    # FAISS index, and parquet files are not committed to Git.
    beit3_faiss_index_path: Path | None = None
    beit3_global_ids_path: Path | None = None
    beit3_video_metadata_path: Path | None = None
    beit3_index_meta_path: Path | None = None
    beit3_checkpoint_path: Path | None = None
    beit3_tokenizer_path: Path | None = None
    beit3_device: str = "cuda"
    beit3_max_seq_len: int = 64

    # Optional overrides when the real global_ids.parquet column names don't
    # match the auto-detected candidates (see BEiT3Retriever._detect_columns).
    beit3_col_vector_id: str | None = None
    beit3_col_video_id: str | None = None
    beit3_col_frame_id: str | None = None
    beit3_col_frame_path: str | None = None
    beit3_col_timestamp: str | None = None
    beit3_col_namespace: str | None = None

    # --- Logging ---
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def get_faiss_index_path(self) -> Path:
        """Return the resolved Faiss index path.

        Falls back to ``src/dict/nw/faiss_index_clip.bin`` when no
        explicit override is provided via the environment.
        """
        if self.faiss_index_path is not None:
            return Path(self.faiss_index_path)
        return self.src_dir / "dict" / "nw" / "faiss_index_clip.bin"

    def get_metadata_path(self) -> Path:
        """Return the resolved metadata JSON path.

        Falls back to ``src/dict/metadata_clip.json``.
        """
        if self.metadata_path is not None:
            return Path(self.metadata_path)
        return self.src_dir / "dict" / "metadata_clip.json"

    def get_keyframes_root(self) -> Path:
        """Return the resolved keyframes root directory.

        Falls back to ``src/data/Keyframes``.
        """
        if self.keyframes_root is not None:
            return Path(self.keyframes_root)
        return self.src_dir / "data" / "Keyframes"

    def get_features_root(self) -> Path:
        """Return the resolved features root directory.

        Falls back to ``src/data/features``.
        """
        if self.features_root is not None:
            return Path(self.features_root)
        return self.src_dir / "data" / "features"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    The settings object is created once and reused for the entire
    application lifetime. Use ``get_settings.cache_clear()`` in tests
    to force re-creation.

    Returns:
        The application Settings instance.
    """
    return Settings()
