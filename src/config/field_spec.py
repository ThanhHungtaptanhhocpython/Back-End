"""Typed metadata for every runtime configuration field.

This registry is the single source of truth shared by:

* the runtime-config store (which keys are secret, which are locked),
* the management API (schema + server-side validation), and
* the Settings UI (field grouping, widget type, help text, ranges).

``key`` is the ``.env`` / environment-variable name (upper-case). ``field``
is the matching :class:`~src.config.settings.Settings` attribute (lower-case).
For all current fields the two are related by ``field == key.lower()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field
from typing import Any

# --- field kinds ---------------------------------------------------------------
BOOL = "bool"
INT = "int"
FLOAT = "float"
STR = "str"
URL = "url"
PATH = "path"
JSON = "json"
JSON_OBJECT = "json_object"
SECRET = "secret"
CSV = "csv"
CHOICE = "choice"

# --- groups (ordered as they appear in the UI) -------------------------------
G_SERVER = "Server"
G_DATA = "Data/Media"
G_ELASTIC = "Elasticsearch"
G_RETRIEVAL = "Retrieval"
G_AI = "AI"
G_AGENT = "Agent/VLM"
G_TRAKE = "TRAKE"
G_QA = "Q&A"
G_CLOUD = "Cloud Assets"
G_LAUNCHER = "Launcher"
G_LOGGING = "Logging"

GROUP_ORDER = [
    G_SERVER, G_DATA, G_ELASTIC, G_RETRIEVAL, G_AI, G_AGENT,
    G_TRAKE, G_QA, G_CLOUD, G_LAUNCHER, G_LOGGING,
]

# One-line orientation shown under each group heading in the UI.
GROUP_HELP = {
    G_SERVER: "Where the API binds and which browser origins may call it.",
    G_DATA: "Local dataset / media paths for keyframes, playback and captures.",
    G_ELASTIC: "Elasticsearch cluster used for OCR / ASR text search.",
    G_RETRIEVAL: "BEiT3 visual-search runtime artifacts (checkpoint, index, parquet).",
    G_AI: "Optional multi-provider gateway and per-provider keys / models.",
    G_AGENT: "Agent Search planner and VLM candidate-verification tuning.",
    G_TRAKE: "Ordered-event (TRAKE) retrieval and scoring parameters.",
    G_QA: "Grounded video Q&A retrieval and confidence tuning.",
    G_CLOUD: "Read the dataset from Azure Blob or S3-compatible storage.",
    G_LAUNCHER: "Behaviour of `python -m launcher` (restart, health, frontend).",
    G_LOGGING: "Log verbosity and (planned) file / request logging.",
}

# Fields shown by default. Everything else -- and anything with no runtime flow
# yet -- is hidden behind the "Advanced" toggle so first-time users see a short,
# meaningful list instead of ~160 knobs.
_BASIC_KEYS = {
    # Server
    "ENV", "DEBUG", "HOST", "PORT", "CORS_ORIGINS", "SRC_DIR",
    # Data / Media
    "KEYFRAMES_ROOT", "MEDIA_INFO_PATH", "MAP_KEYFRAMES_PATH",
    # Elasticsearch
    "ELASTICSEARCH_URL",
    # Retrieval
    "BEIT3_FAISS_INDEX_PATH", "BEIT3_GLOBAL_IDS_PATH", "BEIT3_VIDEO_METADATA_PATH",
    "BEIT3_INDEX_META_PATH", "BEIT3_CHECKPOINT_PATH", "BEIT3_TOKENIZER_PATH", "BEIT3_DEVICE",
    # AI gateway essentials
    "AI_GATEWAY_ENABLED", "AI_TEXT_PRIORITY", "AI_VISION_PRIORITY", "AI_LOCAL_FALLBACK_ENABLED",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "NIM_ENABLED", "NIM_API_KEY", "NIM_TEXT_MODEL", "NIM_VISION_MODEL",
    "CEREBRAS_ENABLED", "CEREBRAS_API_KEY", "CEREBRAS_TEXT_MODEL", "CEREBRAS_VISION_MODEL",
    "GROQ_ENABLED", "GROQ_API_KEY", "GROQ_TEXT_MODEL", "GROQ_VISION_MODEL",
    "OPENROUTER_ENABLED", "OPENROUTER_VISION_MODEL",
    "GEMINI_ENABLED", "GEMINI_API_KEY", "GEMINI_TEXT_MODEL", "GEMINI_VISION_MODEL",
    "CLOUDFLARE_ENABLED", "CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_TEXT_MODEL", "CLOUDFLARE_VISION_MODEL",
    # Agent / VLM top-level toggles
    "AGENT_LLM_ENABLED", "AGENT_LLM_MODEL", "AGENT_VLM_ENABLED", "AGENT_VLM_MODEL",
    # TRAKE / Q&A top-level toggles
    "TRAKE_OCR_ENABLED", "TRAKE_ASR_ENABLED", "TRAKE_VLM_ENABLED", "QA_VLM_ENABLED",
    # Cloud assets
    "CLOUD_ASSETS_ENABLED", "CLOUD_ASSETS_PROVIDER",
    "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_PRIMARY_KEY",
    "S3_ENDPOINT_URL", "S3_REGION", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
    # Launcher
    "LAUNCHER_FRONTEND_ENABLED",
    # Logging
    "LOG_LEVEL",
}

_BOOL_TRUE = {"1", "true", "yes", "on", "y", "t"}
_BOOL_FALSE = {"0", "false", "no", "off", "n", "f", ""}


@dataclass(frozen=True)
class FieldSpec:
    key: str
    group: str
    kind: str
    label: str = ""
    help: str = ""
    secret: bool = False
    locked: bool = False
    has_runtime_flow: bool = True
    choices: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    placeholder: str = ""
    restart_required: bool = True
    # Hide this field in the UI while another field currently equals a value,
    # e.g. {"AI_GATEWAY_ENABLED": "true"} for legacy single-provider knobs.
    hide_when: dict[str, str] | None = None
    field_overrides: dict[str, Any] = _dc_field(default_factory=dict)

    @property
    def field(self) -> str:
        return self.field_overrides.get("field", self.key.lower())

    @property
    def advanced(self) -> bool:
        """Hidden behind the UI's 'Advanced' toggle by default."""
        return self.key not in _BASIC_KEYS or not self.has_runtime_flow

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "field": self.field,
            "group": self.group,
            "kind": self.kind,
            "label": self.label or self.key.replace("_", " ").title(),
            "help": self.help,
            "secret": self.secret,
            "locked": self.locked,
            "advanced": self.advanced,
            "has_runtime_flow": self.has_runtime_flow,
            "choices": list(self.choices) if self.choices else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "placeholder": self.placeholder,
            "restart_required": self.restart_required,
            "hide_when": dict(self.hide_when) if self.hide_when else None,
        }


def _f(key: str, group: str, kind: str, **kw: Any) -> FieldSpec:
    return FieldSpec(key=key, group=group, kind=kind, **kw)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
FIELD_SPECS: tuple[FieldSpec, ...] = (
    # -- Server --------------------------------------------------------------
    _f("ENV", G_SERVER, CHOICE, choices=("development", "production"),
       help="Environment name. 'production' also forces DEBUG off."),
    _f("DEBUG", G_SERVER, BOOL, help="Uvicorn auto-reload / verbose errors."),
    _f("HOST", G_SERVER, STR, placeholder="0.0.0.0",
       help="Bind address for the FastAPI server. Changing it makes the "
            "launcher rebuild the frontend API base URL and reconnect."),
    _f("PORT", G_SERVER, INT, minimum=1, maximum=65535, placeholder="3000",
       help="TCP port for the FastAPI server."),
    _f("CORS_ORIGINS", G_SERVER, CSV,
       placeholder="http://localhost:5173,http://127.0.0.1:5173",
       help="Comma-separated allowed browser origins. '*' disables credentials."),
    _f("SRC_DIR", G_SERVER, PATH, locked=True, restart_required=True,
       help="Absolute path to the backend 'src/' directory. Derived from the "
            "install location and cannot be edited."),

    # -- Data / Media -----------------------------------------------------------
    _f("KEYFRAMES_ROOT", G_DATA, PATH,
       help="Root directory of keyframe images. In cloud-assets mode this is "
            "the local LRU cache that resolve-keyframe fills on demand."),
    _f("FEATURES_ROOT", G_DATA, PATH, has_runtime_flow=False,
       help="Legacy per-video .npy feature root. Kept for offline scripts; "
            "the live search path does not read it."),
    _f("MEDIA_INFO_PATH", G_DATA, PATH,
       help="ZIP or directory of per-video media-info JSON (watch_url + length)."),
    _f("MAP_KEYFRAMES_PATH", G_DATA, PATH,
       help="ZIP or directory of per-video map-keyframes CSV (authoritative FPS)."),
    _f("PLAYBACK_OFFSETS_JSON", G_DATA, JSON_OBJECT,
       placeholder='{"L21_V029": -172}',
       help="Optional video_id -> playback offset seconds. Default offset is 0."),
    _f("VIDEO_CAPTURE_CACHE_PATH", G_DATA, PATH,
       help="Directory for generated WebP stills. Blank -> .cache/video-captures."),
    _f("VIDEO_CAPTURE_FFMPEG_BIN", G_DATA, STR, placeholder="ffmpeg",
       help="FFmpeg binary name or absolute path."),
    _f("VIDEO_CAPTURE_EXTRACT_TIMEOUT_SECONDS", G_DATA, FLOAT, minimum=1, maximum=600,
       help="Wall-clock limit for one yt-dlp + FFmpeg extraction."),
    _f("VIDEO_CAPTURE_CACHE_MAX_BYTES", G_DATA, INT, minimum=1_000_000,
       help="Evict least-recently-used stills once the cache exceeds this size."),

    # -- Elasticsearch --------------------------------------------------------
    _f("ELASTICSEARCH_URL", G_ELASTIC, URL, placeholder="http://localhost:9200",
       help="Base URL of the Elasticsearch cluster used for OCR/ASR search."),

    # -- Retrieval (BEiT3) --------------------------------------------------
    _f("BEIT3_FAISS_INDEX_PATH", G_RETRIEVAL, PATH,
       help="FAISS index file for the BEiT3 visual-search path."),
    _f("BEIT3_GLOBAL_IDS_PATH", G_RETRIEVAL, PATH, help="global_ids.parquet."),
    _f("BEIT3_VIDEO_METADATA_PATH", G_RETRIEVAL, PATH, help="video_metadata.parquet."),
    _f("BEIT3_INDEX_META_PATH", G_RETRIEVAL, PATH, help="index_meta.json."),
    _f("BEIT3_CHECKPOINT_PATH", G_RETRIEVAL, PATH, help="BEiT3 model checkpoint (.pth)."),
    _f("BEIT3_TOKENIZER_PATH", G_RETRIEVAL, PATH, help="BEiT3 SentencePiece model (.spm)."),
    _f("BEIT3_DEVICE", G_RETRIEVAL, CHOICE, choices=("cpu", "cuda"),
       help="Torch device for BEiT3 text encoding."),
    _f("BEIT3_MAX_SEQ_LEN", G_RETRIEVAL, INT, minimum=8, maximum=512),
    _f("BEIT3_COL_VECTOR_ID", G_RETRIEVAL, STR, help="Optional parquet column override."),
    _f("BEIT3_COL_VIDEO_ID", G_RETRIEVAL, STR, help="Optional parquet column override."),
    _f("BEIT3_COL_FRAME_ID", G_RETRIEVAL, STR, help="Optional parquet column override."),
    _f("BEIT3_COL_FRAME_PATH", G_RETRIEVAL, STR, help="Optional parquet column override."),
    _f("BEIT3_COL_TIMESTAMP", G_RETRIEVAL, STR, help="Optional parquet column override."),
    _f("BEIT3_COL_NAMESPACE", G_RETRIEVAL, STR, help="Optional parquet column override."),

    # -- AI: legacy single-provider LLM knobs (used only when the gateway is OFF) --
    _f("LLM_PROVIDER", G_AI, CHOICE, label="Legacy LLM provider",
       choices=("auto", "openai", "openrouter", "anthropic", "nvidia", "google"),
       hide_when={"AI_GATEWAY_ENABLED": "true"},
       help="Chat planner provider selector for the pre-gateway path. Ignored "
            "when AI_GATEWAY_ENABLED is on."),
    _f("OPENAI_API_KEY", G_AI, SECRET, secret=True, label="OpenAI API key",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("OPENAI_MODEL", G_AI, STR, placeholder="gpt-4o-mini", label="OpenAI model",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("OPENROUTER_API_KEY", G_AI, SECRET, secret=True, label="OpenRouter API key",
       help="Used by both the legacy path and the gateway's OpenRouter provider."),
    _f("OPENROUTER_MODEL", G_AI, STR, placeholder="openai/gpt-4o-mini",
       label="OpenRouter text model",
       help="Text model for OpenRouter (legacy path and the gateway's OpenRouter provider)."),
    _f("OPENROUTER_BASE_URL", G_AI, URL, placeholder="https://openrouter.ai/api/v1",
       label="OpenRouter base URL"),
    _f("OPENROUTER_MAX_TOKENS", G_AI, INT, minimum=1, maximum=32768,
       label="OpenRouter max tokens", hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("OPENROUTER_SITE_URL", G_AI, URL, label="OpenRouter HTTP-Referer"),
    _f("OPENROUTER_APP_NAME", G_AI, STR, label="OpenRouter X-Title"),
    _f("OPENROUTER_TRANSLATE_MODEL", G_AI, STR, label="OpenRouter /translate model",
       help="Optional dedicated model for the /translate fallback (legacy path)."),
    _f("OPENROUTER_TRANSLATE_MAX_TOKENS", G_AI, INT, minimum=1, maximum=8192,
       label="OpenRouter /translate max tokens"),
    _f("ANTHROPIC_API_KEY", G_AI, SECRET, secret=True, label="Anthropic API key (legacy)",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("ANTHROPIC_MODEL", G_AI, STR, label="Anthropic model (legacy)",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("ANTHROPIC_MAX_TOKENS", G_AI, INT, minimum=1, maximum=32768,
       label="Anthropic max tokens (legacy)", hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("NVIDIA_API_KEY", G_AI, SECRET, secret=True, label="NVIDIA API key (legacy)",
       help="Pre-gateway NVIDIA path. The gateway's NVIDIA NIM provider uses "
            "NIM_API_KEY instead.",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("NVIDIA_MODEL", G_AI, STR, label="NVIDIA model (legacy)",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("NVIDIA_MAX_TOKENS", G_AI, INT, minimum=1, maximum=32768,
       label="NVIDIA max tokens (legacy)", hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("NVIDIA_TOP_P", G_AI, FLOAT, minimum=0.0, maximum=1.0,
       label="NVIDIA top_p (legacy)", hide_when={"AI_GATEWAY_ENABLED": "true"}),
    _f("GOOGLE_API_KEY", G_AI, SECRET, secret=True, label="Google API key",
       help="Legacy Google path; also the fallback key for the gateway's Gemini "
            "provider when GEMINI_API_KEY is blank."),
    _f("GOOGLE_MODEL", G_AI, STR, label="Google model (legacy)",
       hide_when={"AI_GATEWAY_ENABLED": "true"}),

    # -- AI: multi-provider gateway -------------------------------------------
    _f("AI_GATEWAY_ENABLED", G_AI, BOOL,
       help="Route translation / Agent planner / VLM through the provider "
            "gateway with Text and Vision fallback lists."),
    _f("AI_TEXT_PRIORITY", G_AI, CSV,
       placeholder="nim,cerebras,groq,openrouter,gemini,cloudflare",
       help="Ordered Text fallback list (translation + Agent planner)."),
    _f("AI_VISION_PRIORITY", G_AI, CSV,
       placeholder="gemini,openrouter,nim,cloudflare,groq,cerebras",
       help="Ordered Vision fallback list (Q&A + VLM verifier)."),
    _f("AI_LOCAL_FALLBACK_ENABLED", G_AI, BOOL,
       help="When every Text provider fails, use the local planner/ranking. "
            "When every Vision provider fails, return a non-VLM result."),
    _f("AI_GATEWAY_MAX_TOKENS", G_AI, INT, minimum=1, maximum=32768),

    # NVIDIA NIM (OpenAI-compatible hosted endpoint)
    _f("NIM_ENABLED", G_AI, BOOL),
    _f("NIM_API_KEY", G_AI, SECRET, secret=True),
    _f("NIM_BASE_URL", G_AI, URL, placeholder="https://integrate.api.nvidia.com/v1"),
    _f("NIM_TEXT_MODEL", G_AI, STR, help="Model ID; discover/confirm via Test."),
    _f("NIM_VISION_MODEL", G_AI, STR, help="Vision model ID; confirm via Test."),
    _f("NIM_TIMEOUT_SECONDS", G_AI, FLOAT, minimum=1, maximum=600),
    # Cerebras
    _f("CEREBRAS_ENABLED", G_AI, BOOL),
    _f("CEREBRAS_API_KEY", G_AI, SECRET, secret=True),
    _f("CEREBRAS_BASE_URL", G_AI, URL, placeholder="https://api.cerebras.ai/v1"),
    _f("CEREBRAS_TEXT_MODEL", G_AI, STR),
    _f("CEREBRAS_VISION_MODEL", G_AI, STR),
    _f("CEREBRAS_TIMEOUT_SECONDS", G_AI, FLOAT, minimum=1, maximum=600),
    # Groq
    _f("GROQ_ENABLED", G_AI, BOOL),
    _f("GROQ_API_KEY", G_AI, SECRET, secret=True),
    _f("GROQ_BASE_URL", G_AI, URL, placeholder="https://api.groq.com/openai/v1"),
    _f("GROQ_TEXT_MODEL", G_AI, STR),
    _f("GROQ_VISION_MODEL", G_AI, STR),
    _f("GROQ_TIMEOUT_SECONDS", G_AI, FLOAT, minimum=1, maximum=600),
    # OpenRouter (gateway view -- reuses OPENROUTER_API_KEY / OPENROUTER_BASE_URL)
    _f("OPENROUTER_ENABLED", G_AI, BOOL),
    _f("OPENROUTER_VISION_MODEL", G_AI, STR),
    _f("OPENROUTER_TIMEOUT_SECONDS", G_AI, FLOAT, minimum=1, maximum=600),
    # Gemini AI Studio (OpenAI-compatible)
    _f("GEMINI_ENABLED", G_AI, BOOL),
    _f("GEMINI_API_KEY", G_AI, SECRET, secret=True,
       help="AI Studio key. Blank -> falls back to GOOGLE_API_KEY."),
    _f("GEMINI_BASE_URL", G_AI, URL,
       placeholder="https://generativelanguage.googleapis.com/v1beta/openai"),
    _f("GEMINI_TEXT_MODEL", G_AI, STR),
    _f("GEMINI_VISION_MODEL", G_AI, STR),
    _f("GEMINI_TIMEOUT_SECONDS", G_AI, FLOAT, minimum=1, maximum=600),
    # Cloudflare Workers AI (OpenAI-compatible; needs Account ID)
    _f("CLOUDFLARE_ENABLED", G_AI, BOOL),
    _f("CLOUDFLARE_API_KEY", G_AI, SECRET, secret=True),
    _f("CLOUDFLARE_ACCOUNT_ID", G_AI, STR, help="Cloudflare account ID for the Workers AI URL."),
    _f("CLOUDFLARE_TEXT_MODEL", G_AI, STR),
    _f("CLOUDFLARE_VISION_MODEL", G_AI, STR),
    _f("CLOUDFLARE_TIMEOUT_SECONDS", G_AI, FLOAT, minimum=1, maximum=600),

    # -- Agent / VLM ---------------------------------------------------------
    # -- Agent Search TEXT planner (LLM) --
    _f("AGENT_LLM_ENABLED", G_AGENT, BOOL, label="Planner (LLM) — enable query enrichment",
       help="Let an LLM rewrite the user's description into enriched English "
            "search queries + a checklist before retrieval. Off = deterministic "
            "local planner only."),
    _f("AGENT_LLM_MODEL", G_AGENT, STR, label="Planner model (legacy path only)",
       hide_when={"AI_GATEWAY_ENABLED": "true"},
       help="Model for the planner when AI_GATEWAY_ENABLED is off. With the "
            "gateway on, the model comes from the Text chain providers on the "
            "AI Providers tab — this field is ignored."),
    _f("AGENT_LLM_MAX_TOKENS", G_AGENT, INT, minimum=1, maximum=32768,
       label="Planner response max tokens"),
    _f("AGENT_VISUAL_QUERY_LIMIT", G_AGENT, INT, minimum=1, maximum=8,
       label="Planner visual-query limit",
       help="How many holistic visual queries the planner may emit."),
    # -- Agent Search VISION verifier (VLM) --
    _f("AGENT_VLM_ENABLED", G_AGENT, BOOL, label="Verifier (VLM) — enable frame check",
       help="Score/rerank the top Agent Search candidates with a vision model. "
            "Off = retrieval order is kept as-is."),
    _f("AGENT_VLM_MODEL", G_AGENT, STR, label="Verifier model (legacy path only)",
       hide_when={"AI_GATEWAY_ENABLED": "true"},
       help="Vision model for the verifier when AI_GATEWAY_ENABLED is off. With "
            "the gateway on, the model comes from the Vision chain providers on "
            "the AI Providers tab — this field is ignored."),
    _f("AGENT_VLM_MAX_CANDIDATES", G_AGENT, INT, minimum=1, maximum=100,
       label="Verifier: frames scored per query"),
    _f("AGENT_VLM_CANDIDATE_POOL", G_AGENT, INT, minimum=1, maximum=400,
       label="Verifier: candidate pool size"),
    _f("AGENT_VLM_PER_VIDEO_LIMIT", G_AGENT, INT, minimum=1, maximum=50,
       label="Verifier: max frames per video"),
    _f("AGENT_VLM_BATCH_SIZE", G_AGENT, INT, minimum=1, maximum=32,
       label="Verifier: images per model call"),
    _f("AGENT_VLM_MAX_TOKENS", G_AGENT, INT, minimum=1, maximum=32768,
       label="Verifier: response max tokens"),
    _f("AGENT_VLM_TIMEOUT_SECONDS", G_AGENT, FLOAT, minimum=1, maximum=600,
       label="Verifier: per-call timeout (s)"),
    _f("AGENT_VLM_IMAGE_MAX_SIDE", G_AGENT, INT, minimum=64, maximum=4096,
       label="Verifier: downscale images to (px)"),
    _f("AGENT_VLM_MAX_RETRIES", G_AGENT, INT, minimum=0, maximum=10,
       label="Verifier: retries per batch"),
    _f("AGENT_VLM_RETRY_BACKOFF_SECONDS", G_AGENT, FLOAT, minimum=0, maximum=60,
       label="Verifier: retry backoff (s)"),
    _f("AGENT_VLM_CACHE_ENABLED", G_AGENT, BOOL, label="Verifier: cache verdicts"),
    _f("AGENT_VLM_CACHE_PATH", G_AGENT, PATH, label="Verifier: verdict cache file"),
    _f("AGENT_VLM_CACHE_MAX_ENTRIES", G_AGENT, INT, minimum=1,
       label="Verifier: cache max entries"),
    _f("AGENT_VLM_CACHE_TTL_SECONDS", G_AGENT, INT, minimum=1,
       label="Verifier: cache entry TTL (s)"),
    # legacy os.getenv knobs, now first-class
    _f("KIS_VQA_RERANK", G_AGENT, BOOL, help="Validate top KIS hits with the reranker."),
    _f("KIS_VQA_RERANK_CANDIDATES", G_AGENT, INT, minimum=1, maximum=60),
    _f("KIS_EVENT_RECALL_K", G_AGENT, INT, minimum=1, maximum=1000),
    _f("KIS_VIDEO_RERANK_VIDEOS", G_AGENT, INT, minimum=1, maximum=16),
    _f("KIS_VQA_FRAMES_PER_EVENT", G_AGENT, INT, minimum=1, maximum=4),
    _f("KIS_VQA_THRESHOLD", G_AGENT, FLOAT, minimum=0.0, maximum=1.0),

    # -- TRAKE ------------------------------------------------------------------
    _f("TRAKE_RETRIEVAL_TOP_K", G_TRAKE, INT, minimum=1, maximum=2000),
    _f("TRAKE_CANDIDATES_PER_EVENT_VIDEO", G_TRAKE, INT, minimum=1, maximum=200),
    _f("TRAKE_BEAM_WIDTH", G_TRAKE, INT, minimum=1, maximum=500),
    _f("TRAKE_MIN_EVENT_GAP_SECONDS", G_TRAKE, FLOAT, minimum=0, maximum=3600),
    _f("TRAKE_MAX_EVENT_GAP_SECONDS", G_TRAKE, FLOAT, minimum=0, maximum=7200),
    _f("TRAKE_MAX_SEQUENCE_SPAN_SECONDS", G_TRAKE, FLOAT, minimum=0, maximum=36000),
    _f("TRAKE_TEMPORAL_DECAY", G_TRAKE, FLOAT, minimum=0, maximum=10),
    _f("TRAKE_EVIDENCE_WINDOW_SECONDS", G_TRAKE, FLOAT, minimum=0.1, maximum=600),
    _f("TRAKE_OCR_ENABLED", G_TRAKE, BOOL),
    _f("TRAKE_ASR_ENABLED", G_TRAKE, BOOL),
    _f("TRAKE_VLM_ENABLED", G_TRAKE, BOOL),
    _f("TRAKE_VLM_MAX_SEQUENCES", G_TRAKE, INT, minimum=1, maximum=100),
    _f("TRAKE_ENABLE_VQA", G_TRAKE, BOOL, help="Legacy VQA rerank toggle for TRAKE."),
    _f("TRAKE_VQA_MAX_SEQUENCES", G_TRAKE, INT, minimum=1, maximum=100),

    # -- Q&A -----------------------------------------------------------------
    _f("QA_RETRIEVAL_POOL", G_QA, INT, minimum=1, maximum=400),
    _f("QA_MAX_FRAMES", G_QA, INT, minimum=1, maximum=64),
    _f("QA_PER_VIDEO_LIMIT", G_QA, INT, minimum=1, maximum=50),
    _f("QA_TEXT_EVIDENCE_TOP_K", G_QA, INT, minimum=1, maximum=100),
    _f("QA_EVIDENCE_WINDOW_SECONDS", G_QA, FLOAT, minimum=0.1, maximum=600),
    _f("QA_VLM_ENABLED", G_QA, BOOL),
    _f("QA_MIN_CONFIDENCE", G_QA, FLOAT, minimum=0.0, maximum=1.0),
    _f("QA_MAX_TOKENS", G_QA, INT, minimum=1, maximum=32768),

    # -- Cloud Assets -------------------------------------------------------
    _f("CLOUD_ASSETS_ENABLED", G_CLOUD, BOOL,
       help="Read model / index / parquet / keyframe assets from cloud storage "
            "instead of local paths."),
    _f("CLOUD_ASSETS_PROVIDER", G_CLOUD, CHOICE,
       choices=("local", "azure_blob", "s3_compatible")),
    _f("CLOUD_ASSETS_MANIFEST_KEY", G_CLOUD, STR, placeholder="hcmai-assets.json",
       help="Object key of the versioned manifest inside the metadata container/bucket."),
    _f("CLOUD_ASSETS_CACHE_PATH", G_CLOUD, PATH,
       help="Local cache root for synced artifacts. Blank -> <app-data>/assets-cache."),
    _f("CLOUD_ASSETS_CACHE_MAX_BYTES", G_CLOUD, INT, minimum=0,
       help="Soft cap for synced artifacts (0 = unbounded)."),
    _f("CLOUD_ASSETS_KEYFRAME_CACHE_MAX_BYTES", G_CLOUD, INT, minimum=1_000_000,
       help="LRU cap for on-demand keyframe downloads."),
    # Azure Blob
    _f("AZURE_STORAGE_ACCOUNT_NAME", G_CLOUD, STR),
    _f("AZURE_STORAGE_CONNECTION_STRING", G_CLOUD, SECRET, secret=True),
    _f("AZURE_STORAGE_PRIMARY_KEY", G_CLOUD, SECRET, secret=True),
    _f("AZURE_BLOB_CONTAINER_KEYFRAMES", G_CLOUD, STR, placeholder="keyframes"),
    _f("AZURE_BLOB_CONTAINER_EMBEDDINGS", G_CLOUD, STR, placeholder="embeddings"),
    _f("AZURE_BLOB_CONTAINER_METADATA", G_CLOUD, STR, placeholder="metadata"),
    # S3-compatible
    _f("S3_ENDPOINT_URL", G_CLOUD, URL, placeholder="https://<account>.r2.cloudflarestorage.com"),
    _f("S3_REGION", G_CLOUD, STR, placeholder="auto"),
    _f("S3_BUCKET", G_CLOUD, STR),
    _f("S3_ACCESS_KEY_ID", G_CLOUD, SECRET, secret=True),
    _f("S3_SECRET_ACCESS_KEY", G_CLOUD, SECRET, secret=True),
    _f("S3_METADATA_PREFIX", G_CLOUD, STR, placeholder="metadata/",
       help="Key prefix under which the manifest and metadata objects live."),

    # -- Launcher -----------------------------------------------------------
    _f("LAUNCHER_FRONTEND_ENABLED", G_LAUNCHER, BOOL,
       help="Have the launcher also start / restart the local frontend dev server."),
    _f("LAUNCHER_FRONTEND_DIR", G_LAUNCHER, PATH, placeholder="frontend"),
    _f("LAUNCHER_FRONTEND_PORT", G_LAUNCHER, INT, minimum=1, maximum=65535, placeholder="5173"),
    _f("LAUNCHER_HEALTH_TIMEOUT_SECONDS", G_LAUNCHER, FLOAT, minimum=1, maximum=600,
       help="If the app is not healthy within this window after a restart, the "
            "launcher restores the previous revision."),
    _f("LAUNCHER_HEALTH_POLL_INTERVAL_SECONDS", G_LAUNCHER, FLOAT, minimum=0.1, maximum=30),

    # -- Logging -----------------------------------------------------------
    _f("LOG_LEVEL", G_LOGGING, CHOICE,
       choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")),
    _f("LOG_FILE_PATH", G_LOGGING, PATH, has_runtime_flow=False,
       help="Declared for a future file handler; not wired to the logger yet."),
    _f("LOG_REQUEST_BODY", G_LOGGING, BOOL, has_runtime_flow=False,
       help="Declared for future verbose request logging; not wired yet."),

    # -- Chat --------------------------------------------------------------
    _f("CHAT_HISTORY_MESSAGES", G_SERVER, INT, minimum=0, maximum=100,
       help="Number of prior chat turns kept in agent memory."),
)

# ---------------------------------------------------------------------------
_BY_KEY: dict[str, FieldSpec] = {s.key: s for s in FIELD_SPECS}
_BY_FIELD: dict[str, FieldSpec] = {s.field: s for s in FIELD_SPECS}


def all_specs() -> tuple[FieldSpec, ...]:
    return FIELD_SPECS


def by_key(key: str) -> FieldSpec | None:
    return _BY_KEY.get(key.upper())


def by_field(field_name: str) -> FieldSpec | None:
    return _BY_FIELD.get(field_name.lower())


def secret_keys() -> set[str]:
    return {s.key for s in FIELD_SPECS if s.secret}


def locked_keys() -> set[str]:
    return {s.key for s in FIELD_SPECS if s.locked}


def known_keys() -> set[str]:
    return set(_BY_KEY)


def grouped() -> list[dict[str, Any]]:
    """Specs bucketed by group in UI order (for the management API schema)."""
    out: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        items = [s.to_dict() for s in FIELD_SPECS if s.group == group]
        if items:
            out.append(
                {
                    "group": group,
                    "help": GROUP_HELP.get(group, ""),
                    "basic_count": sum(1 for it in items if not it["advanced"]),
                    "fields": items,
                }
            )
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class ValidationError(ValueError):
    pass


def validate_value(spec: FieldSpec, raw: Any) -> str:
    """Return the normalised string form of ``raw`` for ``spec``.

    Raises :class:`ValidationError` with a human-readable message on failure.
    A blank value is always allowed (means "unset" / "keep existing secret").
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if text == "":
        return ""

    kind = spec.kind
    if kind == BOOL:
        low = text.lower()
        if low in _BOOL_TRUE:
            return "true"
        if low in _BOOL_FALSE:
            return "false"
        raise ValidationError(f"{spec.key}: expected a boolean, got {text!r}")

    if kind == INT:
        try:
            value = int(text, 10)
        except ValueError:
            raise ValidationError(f"{spec.key}: expected an integer, got {text!r}")
        _check_range(spec, value)
        return str(value)

    if kind == FLOAT:
        try:
            value = float(text)
        except ValueError:
            raise ValidationError(f"{spec.key}: expected a number, got {text!r}")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValidationError(f"{spec.key}: number must be finite")
        _check_range(spec, value)
        return repr(value) if value % 1 else str(int(value)) if value.is_integer() else str(value)

    if kind == CHOICE:
        if spec.choices and text not in spec.choices:
            raise ValidationError(
                f"{spec.key}: must be one of {', '.join(spec.choices)}"
            )
        return text

    if kind == URL:
        if not (text.startswith("http://") or text.startswith("https://")):
            raise ValidationError(f"{spec.key}: must be an http(s) URL")
        return text

    if kind in (JSON, JSON_OBJECT):
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise ValidationError(f"{spec.key}: invalid JSON ({exc})")
        if kind == JSON_OBJECT and not isinstance(parsed, dict):
            raise ValidationError(f"{spec.key}: JSON must be an object")
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    if kind == PATH:
        if "\x00" in text:
            raise ValidationError(f"{spec.key}: path contains a null byte")
        return text

    if kind == CSV:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return ",".join(parts)

    # STR / SECRET
    return text


def _check_range(spec: FieldSpec, value: float) -> None:
    if spec.minimum is not None and value < spec.minimum:
        raise ValidationError(f"{spec.key}: must be >= {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise ValidationError(f"{spec.key}: must be <= {spec.maximum}")
