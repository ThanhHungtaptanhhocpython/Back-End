"""Multi-provider AI gateway.

A thin abstraction over OpenAI-compatible chat endpoints (NVIDIA NIM, Cerebras,
Groq, OpenRouter, Gemini AI Studio, Cloudflare Workers AI) with two independent
ordered fallback chains -- Text (translation / Agent planner) and Vision (Q&A /
VLM verifier).

When ``AI_GATEWAY_ENABLED`` is off, callers keep their existing single-provider
behaviour untouched.
"""

from src.services.ai.base import (  # noqa: F401
    RETRYABLE_CATEGORIES,
    AllProvidersFailed,
    ChatResult,
    ProviderAttempt,
    ProviderError,
)
from src.services.ai.gateway import (  # noqa: F401
    text_completion,
    vision_available,
    vision_completion,
)
from src.services.ai.registry import (  # noqa: F401
    all_provider_ids,
    build_provider,
    provider_status,
    text_chain,
    vision_chain,
)
