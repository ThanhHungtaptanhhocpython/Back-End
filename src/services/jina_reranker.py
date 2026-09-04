"""Cloud Jina multimodal reranking for standard KIS results.

Jina CLIP remains the first-stage retriever.  This module only reorders a
small candidate pool with ``jina-reranker-m0`` and deliberately falls back to
the original retrieval order when the optional cloud API is unavailable.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from src.config.settings import get_settings
from src.services.openrouter_vlm_verifier import resolve_keyframe_path

logger = logging.getLogger(__name__)


def _image_to_data_url(path: Path, max_side: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _post_rerank(payload: dict[str, Any], *, api_key: str, endpoint: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def rerank_kis_results(query: str, results: list[dict[str, Any]], *, settings: Any = None) -> list[dict[str, Any]]:
    """Rerank KIS candidates using Jina's multimodal cloud API.

    Result scores become Jina reranker relevance scores.  The original CLIP
    score is retained as ``retrieval_score`` so callers can inspect both
    stages.  Failures return the unmodified retrieval ranking.
    """
    settings = settings or get_settings()
    if not results or not bool(getattr(settings, "jina_reranker_enabled", False)):
        return results

    api_key = str(getattr(settings, "jina_reranker_api_key", "") or "").strip()
    if not api_key:
        logger.warning("Jina KIS reranker is enabled but JINA_RERANKER_API_KEY is not configured.")
        return results

    pool_size = max(1, min(int(getattr(settings, "jina_reranker_candidate_pool", 20)), len(results), 100))
    max_side = max(128, min(int(getattr(settings, "jina_reranker_image_max_side", 768)), 1600))
    candidates: list[tuple[int, dict[str, Any], Path]] = []
    for original_index, item in enumerate(results[:pool_size]):
        image_path = resolve_keyframe_path(item)
        if image_path is not None:
            candidates.append((original_index, item, image_path))

    if not candidates:
        logger.warning("Jina KIS reranker skipped: no candidate keyframes could be resolved.")
        return results

    try:
        payload = {
            "model": str(getattr(settings, "jina_reranker_model", "jina-reranker-m0")),
            "query": query,
            "top_n": len(candidates),
            "documents": [{"image": _image_to_data_url(path, max_side)} for _, _, path in candidates],
        }
        response = _post_rerank(
            payload,
            api_key=api_key,
            endpoint=str(getattr(settings, "jina_reranker_base_url", "https://api.jina.ai/v1/rerank")),
            timeout=float(getattr(settings, "jina_reranker_timeout_seconds", 45.0)),
        )
        reranked = response.get("results")
        if not isinstance(reranked, list):
            raise ValueError("Jina reranker response does not contain a results list.")
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Jina KIS reranker failed; using Jina CLIP ranking: %s", exc)
        return results

    scored: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for entry in reranked:
        if not isinstance(entry, dict):
            continue
        try:
            candidate_index = int(entry["index"])
            relevance = float(entry["relevance_score"])
            original_index, item, _path = candidates[candidate_index]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if original_index in seen_indices:
            continue
        seen_indices.add(original_index)
        enriched = dict(item)
        enriched["retrieval_score"] = item.get("score")
        enriched["reranker_score"] = relevance
        enriched["reranker_model"] = payload["model"]
        enriched["score"] = relevance
        scored.append(enriched)

    if not scored:
        logger.warning("Jina KIS reranker returned no usable scores; using Jina CLIP ranking.")
        return results

    # Keep candidates missing from the API response and all results outside the
    # pool after the reranked segment, preserving their original recall order.
    remaining = [item for index, item in enumerate(results) if index not in seen_indices]
    merged = scored + remaining
    for rank, item in enumerate(merged, start=1):
        item["rank"] = rank
    return merged
