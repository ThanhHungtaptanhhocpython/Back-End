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

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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
        keyframes_root: Root directory for keyframe images.
        log_level: Python logging level string.
    """

    # --- Server ---
    env: str = "development"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 3000

    # --- Data Paths (resolved relative to src_dir if not absolute) ---
    src_dir: Path = _default_src_dir()
    keyframes_root: Path | None = None

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

    # --- Legacy os.getenv knobs, promoted to first-class settings ---
    kis_vqa_rerank: bool = True
    kis_vqa_rerank_candidates: int = 24
    kis_event_recall_k: int = 300
    kis_video_rerank_videos: int = 8
    kis_vqa_frames_per_event: int = 2
    kis_vqa_threshold: float = 0.55
    trake_enable_vqa: bool = False
    trake_vqa_max_sequences: int = 5

    # --- Multi-provider AI gateway ---
    # When enabled, translation / Agent planner / VLM verifier route through the
    # provider gateway, trying each entry of the Text or Vision priority list in
    # order and falling back to local behaviour when every provider fails.
    ai_gateway_enabled: bool = False
    ai_text_priority: str = "nim,cerebras,groq,openrouter,kilo,gemini,cloudflare"
    ai_vision_priority: str = "gemini,openrouter,kilo,nim,cloudflare,groq,cerebras"
    ai_local_fallback_enabled: bool = True
    ai_gateway_max_tokens: int = 1024

    nim_enabled: bool = False
    nim_api_key: str | None = None
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_text_model: str = ""
    nim_vision_model: str = ""
    nim_timeout_seconds: float = 45.0

    cerebras_enabled: bool = False
    cerebras_api_key: str | None = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_text_model: str = ""
    cerebras_vision_model: str = ""
    cerebras_timeout_seconds: float = 45.0

    groq_enabled: bool = False
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_text_model: str = ""
    groq_vision_model: str = ""
    groq_timeout_seconds: float = 45.0

    openrouter_enabled: bool = False
    openrouter_vision_model: str = ""
    openrouter_timeout_seconds: float = 45.0

    gemini_enabled: bool = False
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_text_model: str = ""
    gemini_vision_model: str = ""
    gemini_timeout_seconds: float = 45.0

    cloudflare_enabled: bool = False
    cloudflare_api_key: str | None = None
    cloudflare_account_id: str = ""
    cloudflare_text_model: str = ""
    cloudflare_vision_model: str = ""
    cloudflare_timeout_seconds: float = 45.0

    # Kilo AI Gateway (OpenRouter-compatible aggregator, kilocode.ai)
    kilo_enabled: bool = False
    kilo_api_key: str | None = None
    kilo_base_url: str = "https://kilocode.ai/api/openrouter"
    kilo_text_model: str = ""
    kilo_vision_model: str = ""
    kilo_timeout_seconds: float = 45.0

    # --- Cloud asset storage (dataset read from Azure Blob or S3-compatible) ---
    cloud_assets_enabled: bool = False
    cloud_assets_provider: str = "local"  # local | azure_blob | s3_compatible
    cloud_assets_manifest_key: str = "hcmai-assets.json"
    cloud_assets_cache_path: Path | None = None
    cloud_assets_keyframe_cache_max_bytes: int = 5 * 1024 * 1024 * 1024

    azure_storage_account_name: str = ""
    azure_storage_connection_string: str | None = None
    azure_storage_primary_key: str | None = None
    azure_blob_container_keyframes: str = "keyframes"
    azure_blob_container_embeddings: str = "embeddings"
    azure_blob_container_metadata: str = "metadata"

    s3_endpoint_url: str = ""
    s3_region: str = "auto"
    s3_bucket: str = ""
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_metadata_prefix: str = "metadata/"

    # --- Local runtime launcher ---
    launcher_frontend_enabled: bool = False
    launcher_frontend_dir: Path | None = None
    launcher_frontend_port: int = 5173
    launcher_health_timeout_seconds: float = 60.0
    launcher_health_poll_interval_seconds: float = 1.0

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
        "keyframes_root",
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
        "cloud_assets_cache_path",
        "launcher_frontend_dir",
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
        "nim_api_key",
        "cerebras_api_key",
        "groq_api_key",
        "gemini_api_key",
        "cloudflare_api_key",
        "kilo_api_key",
        "azure_storage_connection_string",
        "azure_storage_primary_key",
        "s3_access_key_id",
        "s3_secret_access_key",
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

    def get_keyframes_root(self) -> Path:
        """Return the resolved keyframes root directory.

        Falls back to ``src/data/Keyframes``.
        """
        if self.keyframes_root is not None:
            return Path(self.keyframes_root)
        return self.src_dir / "data" / "Keyframes"

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

    def get_cloud_assets_cache_path(self) -> Path:
        """Root directory for synced cloud artifacts / on-demand keyframes.

        Falls back to ``<app-data>/assets-cache``.
        """
        if self.cloud_assets_cache_path is not None:
            return Path(self.cloud_assets_cache_path)
        from src.config.app_paths import get_assets_cache_dir

        return get_assets_cache_dir()

    def get_launcher_frontend_dir(self) -> Path:
        """Directory of the local frontend the launcher starts. Falls back to
        ``<repo>/frontend``."""
        if self.launcher_frontend_dir is not None:
            return Path(self.launcher_frontend_dir)
        return self.src_dir.parent / "frontend"

    def get_ai_text_priority(self) -> list[str]:
        return [p.strip().lower() for p in self.ai_text_priority.split(",") if p.strip()]

    def get_ai_vision_priority(self) -> list[str]:
        return [p.strip().lower() for p in self.ai_vision_priority.split(",") if p.strip()]

    def redacted_runtime_values(self) -> dict[str, str]:
        """Every registered field as a string, with secrets reduced to a marker.

        Used by the management API so the UI can render the current runtime
        configuration without ever receiving an API key or connection string.
        """
        from src.config import field_spec

        out: dict[str, str] = {}
        for spec in field_spec.all_specs():
            value = getattr(self, spec.field, None)
            if spec.secret:
                out[spec.key] = "********" if value else ""
                continue
            if value is None:
                out[spec.key] = ""
            elif isinstance(value, bool):
                out[spec.key] = "true" if value else "false"
            else:
                out[spec.key] = str(value)
        return out


# Field kinds for which an empty override string is meaningless: fall back to
# the code default instead of feeding "" into Pydantic type coercion.
_SKIP_EMPTY_KINDS = {"bool", "int", "float", "choice"}


def _build_settings() -> Settings:
    """Construct :class:`Settings` layering the active store revision on top of
    the code defaults (``.env`` is bypassed once the store is authoritative)."""
    try:
        from src.config.runtime_store import load_effective_overrides

        overrides = load_effective_overrides()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Runtime config store unavailable, using .env only: %s", exc)
        overrides = {}

    if not overrides:
        return Settings()

    from src.config import field_spec

    kwargs: dict[str, object] = {}
    for key, value in overrides.items():
        spec = field_spec.by_key(key)
        if spec is not None and value == "" and spec.kind in _SKIP_EMPTY_KINDS:
            continue
        field_name = spec.field if spec is not None else key.lower()
        kwargs[field_name] = value
    return Settings(_env_file=None, **kwargs)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Precedence (lowest to highest): code defaults < ``.env`` / environment <
    the active SQLite runtime-config revision. Once the store holds an active
    revision it is authoritative and ``.env`` is no longer read.

    Use ``get_settings.cache_clear()`` in tests to force re-creation.
    """
    return _build_settings()
