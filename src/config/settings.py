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

from pydantic import field_validator
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

    # --- Video playback / frame capture ---
    # Archive (or directory) of per-video media-info JSON files carrying the
    # YouTube ``watch_url`` and ``length``. ``media-info-aic25-b1.zip`` is a
    # runtime asset that is not committed to Git; point this at wherever it
    # lives on the deployment machine.
    media_info_path: Path | None = None
    # Archive (or directory) of per-video map-keyframes CSV files. These carry
    # the authoritative per-video FPS and the keyframe-ordinal -> frame index
    # mapping. Defaults to the extracted ``src/dict/map-keyframes`` directory;
    # an existing ``src/dict/map-keyframes.zip`` remains supported as fallback.
    map_keyframes_path: Path | None = None
    # Optional JSON object mapping ``video_id`` -> playback offset in seconds
    # (``source_time = playback_time - playback_offset``). Only use this for
    # videos whose YouTube timeline has been verified to differ from the
    # dataset timeline; the default offset is ``0`` for every video.
    playback_offsets_json: str = ""

    # --- Captured-frame previews (exact server-side still extraction) ---
    # Directory that stores generated WebP stills at
    # ``<cache>/<video_id>/<frame_idx>.webp``. Defaults to
    # ``.cache/video-captures`` under the repository root. Only preview
    # extraction depends on the tools below; the app starts fine without them.
    video_capture_cache_path: Path | None = None
    # Name of (or absolute path to) the FFmpeg binary used to decode a single
    # frame from the resolved source stream.
    video_capture_ffmpeg_bin: str = "ffmpeg"
    # Hard wall-clock limit (seconds) for one yt-dlp + FFmpeg extraction.
    video_capture_extract_timeout_seconds: float = 90.0
    # Evict least-recently-used stills once the cache exceeds this many bytes.
    video_capture_cache_max_bytes: int = 500 * 1024 * 1024

    # --- External Services ---
    elasticsearch_url: str = "http://localhost:9200"

    # --- CORS ---
    # Comma-separated list of allowed origins. Use "*" to allow any origin
    # (credentials will be disabled automatically in that case).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"

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

    # --- LLM / Agent (chat planner, translation fallback) ---
    llm_provider: str = "auto"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_max_tokens: int = 512
    openrouter_site_url: str = "http://localhost:3000"
    openrouter_app_name: str = "AIC Backend"
    openrouter_translate_model: str | None = None
    openrouter_translate_max_tokens: int = 384
    agent_llm_enabled: bool = False
    agent_llm_model: str | None = None
    agent_llm_max_tokens: int = 900
    agent_visual_query_limit: int = 1
    agent_vlm_enabled: bool = False
    agent_vlm_model: str = "google/gemini-2.5-flash"
    agent_vlm_max_candidates: int = 12
    agent_vlm_candidate_pool: int = 40
    agent_vlm_per_video_limit: int = 3
    agent_vlm_batch_size: int = 4
    agent_vlm_max_tokens: int = 900
    agent_vlm_timeout_seconds: float = 45.0
    agent_vlm_image_max_side: int = 768
    agent_vlm_max_retries: int = 1
    agent_vlm_retry_backoff_seconds: float = 0.5
    agent_vlm_cache_enabled: bool = True
    agent_vlm_cache_path: Path | None = None
    agent_vlm_cache_max_entries: int = 5000
    agent_vlm_cache_ttl_seconds: int = 2592000
    trake_retrieval_top_k: int = 120
    trake_candidates_per_event_video: int = 12
    trake_beam_width: int = 40
    trake_min_event_gap_seconds: float = 0.0
    trake_max_event_gap_seconds: float = 300.0
    trake_max_sequence_span_seconds: float = 900.0
    trake_temporal_decay: float = 0.01
    trake_evidence_window_seconds: float = 12.0
    trake_ocr_enabled: bool = True
    trake_asr_enabled: bool = True
    trake_vlm_enabled: bool = True
    trake_vlm_max_sequences: int = 5
    qa_retrieval_pool: int = 40
    qa_max_frames: int = 8
    qa_per_video_limit: int = 3
    qa_text_evidence_top_k: int = 12
    qa_evidence_window_seconds: float = 15.0
    qa_vlm_enabled: bool = True
    qa_min_confidence: float = 0.55
    qa_max_tokens: int = 700
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20240620"
    anthropic_max_tokens: int = 2048
    nvidia_api_key: str | None = None
    nvidia_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nvidia_max_tokens: int = 2048
    nvidia_top_p: float = 1.0
    google_api_key: str | None = None
    google_model: str = "gemini-1.5-flash"

    # --- Chat memory ---
    chat_history_messages: int = 6

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug_bool(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"release", "production", "prod"}:
            return False
        return value
    @field_validator(
        "faiss_index_path",
        "metadata_path",
        "keyframes_root",
        "features_root",
        "media_info_path",
        "map_keyframes_path",
        "video_capture_cache_path",
        "beit3_faiss_index_path",
        "beit3_global_ids_path",
        "beit3_video_metadata_path",
        "beit3_index_meta_path",
        "beit3_checkpoint_path",
        "beit3_tokenizer_path",
        "agent_vlm_cache_path",
        mode="before",
    )
    @classmethod
    def _blank_path_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "openai_api_key",
        "openrouter_api_key",
        "anthropic_api_key",
        "nvidia_api_key",
        "google_api_key",
        "openrouter_translate_model",
        "agent_llm_model",
        mode="before",
    )
    @classmethod
    def _blank_str_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def get_cors_origins(self) -> list[str]:
        """Return the parsed list of allowed CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def get_faiss_index_path(self) -> Path:
        """Return the resolved Faiss index path.

        Falls back to ``src/dict/faiss_index.bin`` when no
        explicit override is provided via the environment.
        """
        if self.faiss_index_path is not None:
            return Path(self.faiss_index_path)
        return self.src_dir / "dict" / "faiss_index.bin"

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

    def get_media_info_path(self) -> Path:
        """Return the resolved media-info archive/directory path.

        Falls back to ``media-info-aic25-b1.zip`` at the repository root, then
        to the extracted ``src/dict/media-info`` directory.
        """
        if self.media_info_path is not None:
            return Path(self.media_info_path)
        repo_root = self.src_dir.parent
        root_zip = repo_root / "media-info-aic25-b1.zip"
        if root_zip.exists():
            return root_zip
        return self.src_dir / "dict" / "media-info"

    def get_map_keyframes_path(self) -> Path:
        """Return the resolved map-keyframes archive/directory path.

        Falls back to the extracted ``src/dict/map-keyframes`` directory, then
        to an existing ``src/dict/map-keyframes.zip`` archive.
        """
        if self.map_keyframes_path is not None:
            return Path(self.map_keyframes_path)
        map_dir = self.src_dir / "dict" / "map-keyframes"
        if map_dir.exists():
            return map_dir
        bundled_zip = self.src_dir / "dict" / "map-keyframes.zip"
        if bundled_zip.exists():
            return bundled_zip
        return map_dir

    def get_playback_offsets(self) -> dict[str, float]:
        """Return the parsed ``video_id -> offset seconds`` override map.

        An empty or malformed value yields an empty map, which means every
        video uses the default offset of ``0``.
        """
        import json

        raw = (self.playback_offsets_json or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        offsets: dict[str, float] = {}
        for key, value in parsed.items():
            try:
                offsets[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return offsets

    def get_video_capture_cache_path(self) -> Path:
        """Return the directory that stores generated captured-frame stills.

        Falls back to ``.cache/video-captures`` under the repository root.
        """
        if self.video_capture_cache_path is not None:
            return Path(self.video_capture_cache_path)
        return self.src_dir.parent / ".cache" / "video-captures"

    def get_agent_vlm_cache_path(self) -> Path:
        """Return the runtime cache path for OpenRouter VLM verdicts."""
        if self.agent_vlm_cache_path is not None:
            return Path(self.agent_vlm_cache_path)
        return self.src_dir.parent / ".cache" / "agent_vlm_verdicts.json"


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
