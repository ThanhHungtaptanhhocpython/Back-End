"""Fusion Service.

Handles the merging and min-max normalization of scores across 
multiple modalities (Visual, OCR, ASR) to produce a single ranked list.
"""

import logging
from typing import List, Dict, Any

from src.services.user_service import get_elastic_processor
from src.services.reranker_service import reranker_service
from src.utils.nlp_processing import QueryPlanner
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# When the active visual backend is NOT BEiT3, OCR/ASR evidence cannot be
# joined to visual candidates by FAISS integer id: `faiss_id` / `nearest_faiss_id`
# on the ES rows were precomputed against the BEiT3 index, so a numeric
# collision with a Jina `vector_id` would attach unrelated text to a frame.
# The only embedding-space-agnostic key both sides share is
# (video_id, timestamp). This is the half-window, in seconds, a text hit may
# be from a keyframe and still count as evidence for it.
_TEXT_EVIDENCE_WINDOW_S = 2.5


def _active_backend() -> str:
    """The active retrieval backend (see
    ``retrieval_backend.active_backend`` -- Jina by default, forced when cloud
    assets are on). Used only as a fallback when a visual result row does not
    carry ``retrieval_backend``."""
    try:
        from src.services.retrieval_backend import active_backend

        return active_backend()
    except Exception:  # noqa: BLE001
        return "jina_clip_v2"


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _text_evidence_timestamp(row: Dict[str, Any], modality: str) -> float | None:
    """Best timestamp (seconds) for an OCR/ASR ES row, or None if it has none.

    Mirrors grounded_qa_service._evidence_timestamp but stays nullable -- a
    row with no usable timestamp is simply not matched to any keyframe rather
    than being pinned to 0s.
    """
    if modality == "asr":
        near = _finite_or_none(row.get("nearest_timestamp"))
        if near is not None:
            return near
        start = _finite_or_none(row.get("start_time"))
        end = _finite_or_none(row.get("end_time"))
        if start is not None and end is not None:
            return (start + end) / 2.0
        return start if start is not None else end
    return _finite_or_none(row.get("timestamp"))


def merge_by_video_timestamp(
    visual_items: List[Dict[str, Any]],
    ocr_items: List[Dict[str, Any]],
    asr_items: List[Dict[str, Any]],
    weights: Dict[str, float],
    *,
    window_s: float = _TEXT_EVIDENCE_WINDOW_S,
) -> List[Dict[str, Any]]:
    """Fuse multimodal results without ever using a raw FAISS integer as a
    cross-modal key.

    Visual candidates seed the result set; OCR/ASR hits only *augment* a
    visual candidate they share a ``(video_id, timestamp +/- window)`` match
    with. A text hit never creates a new result row (it has no keyframe in
    this backend's space) and a text hit with no usable timestamp is dropped
    rather than blindly attached.
    """
    merged: List[Dict[str, Any]] = []
    by_video: Dict[str, List[Dict[str, Any]]] = {}

    for item in visual_items:
        video_id = str(item.get("video_id") or item.get("video_key") or "")
        doc = {
            "vector_id": item.get("vector_id"),
            "video_id": video_id,
            "frame_id": item.get("frame_id"),
            "frame_idx": item.get("frame_idx"),
            "frame_name": item.get("frame_name", ""),
            "frame_path": item.get("frame_path"),
            "asset_key": item.get("asset_key"),
            "timestamp": item.get("timestamp"),
            "namespace": item.get("namespace") or item.get("split"),
            "split": item.get("split") or item.get("namespace"),
            "folder_key": item.get("folder_key") or item.get("namespace") or item.get("split"),
            "retrieval_backend": item.get("retrieval_backend"),
            "score_breakdown": {"visual": item.get("normalized_score", 0.0), "ocr": 0.0, "asr": 0.0},
        }
        if "image" in item:
            doc["image"] = item["image"]
        for key in ("fps", "source_frame_idx", "keyframe_number", "keyframe_ordinal", "link", "timecode"):
            if item.get(key) not in (None, ""):
                doc[key] = item[key]
        merged.append(doc)
        if video_id:
            by_video.setdefault(video_id, []).append(doc)

    def _augment(items: List[Dict[str, Any]], modality: str) -> None:
        for row in items:
            video_id = str(row.get("video_id") or "")
            ts = _text_evidence_timestamp(row, modality)
            if not video_id or ts is None:
                continue  # no embedding-space-agnostic key -> cannot fuse safely
            score = row.get("normalized_score", 0.0)
            text = row.get("text", "") if modality == "asr" else row.get("ocr_text", "")
            for doc in by_video.get(video_id, ()):
                doc_ts = _finite_or_none(doc.get("timestamp"))
                if doc_ts is None or abs(doc_ts - ts) > window_s:
                    continue
                bd = doc["score_breakdown"]
                bd[modality] = max(bd.get(modality, 0.0), score)
                if text:
                    doc.setdefault(f"{modality}_text", text)

    _augment(ocr_items, "ocr")
    _augment(asr_items, "asr")

    v_w = weights.get("visual", 1.0)
    o_w = weights.get("ocr", 0.0)
    a_w = weights.get("asr", 0.0)
    for doc in merged:
        bd = doc["score_breakdown"]
        doc["final_score"] = (v_w * bd["visual"]) + (o_w * bd["ocr"]) + (a_w * bd["asr"])

    merged.sort(key=lambda d: d["final_score"], reverse=True)
    return merged

def normalize_scores(items: List[Dict[str, Any]], score_key: str = "_score") -> List[Dict[str, Any]]:
    """Apply min-max normalization to a list of results in-place.
    
    Formula: (score - min_score) / (max_score - min_score + epsilon)
    
    Args:
        items: List of result dictionaries containing `score_key`.
        score_key: The dictionary key holding the raw score.
        
    Returns:
        The updated list of dictionaries with a new `normalized_score` key.
    """
    if not items:
        return []
        
    scores = [item.get(score_key, 0.0) for item in items]
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero if all scores are identical
    denominator = max_score - min_score
    if denominator == 0:
        denominator = 1e-9
        
    for item in items:
        raw = item.get(score_key, 0.0)
        norm = (raw - min_score) / denominator
        # Ensure it stays within [0, 1] bounds (due to float precision issues)
        item["normalized_score"] = max(0.0, min(1.0, norm))
        
    return items


def reciprocal_rank_fusion(lists_of_results: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Combine search results from different models using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF_Score = sum(1 / (k + rank))
    
    Args:
        lists_of_results: A list containing multiple result lists (e.g. [visual_results, ocr_results, asr_results]).
        k: Smoothing constant (typically 60).
        
    Returns:
        A single list of merged dictionaries sorted by their rrf_score.
    """
    rrf_scores: Dict[int, float] = {}
    docs_map: Dict[int, Dict[str, Any]] = {}
    
    for result_list in lists_of_results:
        for rank, item in enumerate(result_list, 1):
            fid = item.get("faiss_id")
            if fid is None:
                continue
                
            if fid not in rrf_scores:
                rrf_scores[fid] = 0.0
                docs_map[fid] = item.copy()
                
            rrf_scores[fid] += 1.0 / (k + rank)
            
    final_results = []
    for fid, score in rrf_scores.items():
        doc = docs_map[fid]
        doc["rrf_score"] = score
        final_results.append(doc)
        
    final_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return final_results



def merge_and_rank(
    visual_items: List[Dict[str, Any]],
    ocr_items: List[Dict[str, Any]],
    asr_items: List[Dict[str, Any]],
    weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Merge multimodal results by faiss_id and calculate the final weighted score.
    
    Args:
        visual_items: Normalized results from Faiss (expecting `faiss_id` and `normalized_score`).
        ocr_items: Normalized results from ES OCR (expecting `faiss_id` and `normalized_score`).
        asr_items: Normalized results from ES ASR (expecting `nearest_faiss_id` and `normalized_score`).
        weights: Dictionary like {"visual": 0.6, "ocr": 0.2, "asr": 0.2}.
        
    Returns:
        A sorted list of merged dictionaries containing `score_breakdown` and `final_score`.
    """
    merged: Dict[int, Dict[str, Any]] = {}

    def init_or_get_merged(fid: int, base_doc: dict):
        if fid not in merged:
            merged[fid] = {
                "faiss_id": fid,
                "video_id": base_doc.get("video_id", ""),
                "frame_id": base_doc.get("frame_id"),
                "frame_idx": base_doc.get("frame_idx"),
                "frame_name": base_doc.get("frame_name", ""),
                "frame_path": base_doc.get("frame_path"),
                "timestamp": base_doc.get("timestamp", 0.0),
                "namespace": base_doc.get("namespace"),
                "folder_key": base_doc.get("folder_key") or base_doc.get("namespace"),
                "score_breakdown": {
                    "visual": 0.0,
                    "ocr": 0.0,
                    "asr": 0.0
                }
            }
        return merged[fid]

    # Process Visual
    for item in visual_items:
        fid = item.get("faiss_id")
        if fid is None:
            continue
        doc = init_or_get_merged(fid, item)
        doc["score_breakdown"]["visual"] = item.get("normalized_score", 0.0)
        
        # Capture base64 image if Faiss returned it
        if "image" in item:
            doc["image"] = item["image"]
        for key in ("frame_id", "frame_idx", "frame_path", "namespace", "folder_key", "fps", "source_frame_idx", "keyframe_number"):
            if item.get(key) not in (None, ""):
                doc[key] = item[key]

    # Process OCR
    for item in ocr_items:
        fid = item.get("faiss_id")
        if fid is None:
            continue
        doc = init_or_get_merged(fid, item)
        doc["score_breakdown"]["ocr"] = item.get("normalized_score", 0.0)
        doc["ocr_text"] = item.get("ocr_text", "")

    # Process ASR (uses nearest_faiss_id)
    for item in asr_items:
        fid = item.get("nearest_faiss_id")
        if fid is None:
            continue
        doc = init_or_get_merged(fid, item)
        doc["score_breakdown"]["asr"] = item.get("normalized_score", 0.0)
        doc["asr_text"] = item.get("text", "")

    # Calculate final scores and format output
    final_results = []
    
    v_weight = weights.get("visual", 1.0)
    o_weight = weights.get("ocr", 0.0)
    a_weight = weights.get("asr", 0.0)
    
    for fid, doc in merged.items():
        v_score = doc["score_breakdown"]["visual"]
        o_score = doc["score_breakdown"]["ocr"]
        a_score = doc["score_breakdown"]["asr"]
        
        final_score = (v_weight * v_score) + (o_weight * o_score) + (a_weight * a_score)
        doc["final_score"] = final_score
        
        final_results.append(doc)
        
    # Sort descending by final score
    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results


def multimodal_search(
    visual_query: str, 
    ocr_query: str, 
    asr_query: str, 
    weights: Dict[str, float], 
    topk: int = 100,
    original_query: str = ""
) -> List[Dict[str, Any]]:
    """Execute queries across all backends and fuse the results."""
    
    visual_results = []
    ocr_results = []
    asr_results = []
    visual_backend = "beit3"

    # 1. Fetch from Faiss (Visual)
    if visual_query and weights.get("visual", 0.0) > 0:
        logger.info(f"Executing Visual Search for '{visual_query}'...")
        try:
            # We must import inside the function to avoid circular imports
            # if user_service imports from fusion_service later.
            from src.services.user_service import getImageDataSingleTextSearch
            # getImageDataSingleTextSearch returns the active retrieval
            # backend's results (RETRIEVAL_BACKEND: BEiT3 or Jina CLIP v2, see
            # src/services/retrieval_backend.py) with real FAISS inner-product
            # scores in 'score' and that backend's own vector id in
            # 'vector_id'.
            faiss_list = getImageDataSingleTextSearch(visual_query, topk)
            if faiss_list:
                visual_backend = str(faiss_list[0].get("retrieval_backend") or "").strip().lower() or _active_backend()
            for item in faiss_list:
                item["_score"] = item.get("score", 0.0)
                # Only BEiT3 visual results share a vector-id space with the
                # OCR/ASR `faiss_id` / `nearest_faiss_id` fields. For any other
                # backend, exposing `vector_id` as `faiss_id` here would let a
                # numeric collision cross-contaminate the fusion -- so it is
                # left unset and fusion falls to (video_id, timestamp) below.
                if visual_backend == "beit3":
                    item["faiss_id"] = item.get("vector_id")
            visual_results = normalize_scores(faiss_list)
        except Exception as e:
            logger.error(f"Visual search failed: {e}")

    # 2. Fetch from Elasticsearch (OCR)
    if ocr_query and weights.get("ocr", 0.0) > 0:
        logger.info(f"Executing OCR Search for '{ocr_query}'...")
        try:
            raw_ocr = get_elastic_processor().search_ocr(ocr_query, topk=topk)
            ocr_results = normalize_scores(raw_ocr)
        except Exception as e:
            logger.error(f"OCR search failed: {e}")

    # 3. Fetch from Elasticsearch (ASR)
    if asr_query and weights.get("asr", 0.0) > 0:
        logger.info(f"Executing ASR Search for '{asr_query}'...")
        try:
            raw_asr = get_elastic_processor().search_asr(asr_query, topk=topk)
            asr_results = normalize_scores(raw_asr)
        except Exception as e:
            logger.error(f"ASR search failed: {e}")

    # 4. Merge and rank. BEiT3 keeps the historical FAISS-id join; any other
    #    backend fuses OCR/ASR onto visual candidates strictly by
    #    (video_id, timestamp) so a Jina vector id can never be matched
    #    against a BEiT3-derived OCR/ASR id.
    if visual_backend == "beit3":
        initial_results = merge_and_rank(visual_results, ocr_results, asr_results, weights)
    else:
        logger.info("Multimodal fusion: %s backend -> (video_id, timestamp) join", visual_backend)
        initial_results = merge_by_video_timestamp(visual_results, ocr_results, asr_results, weights)

    # 5. Reranking (Phase 5)
    # We only rerank if there is a visual_query or original_query to ask about
    query_to_ask = original_query or visual_query
    if query_to_ask and initial_results:
        logger.info("Starting VQA Reranking on Top 30 results...")
        top_n = min(30, len(initial_results))
        rerank_pool = initial_results[:top_n]
        bottom_pool = initial_results[top_n:]
        
        vqa_question = QueryPlanner.generate_vqa_question(query_to_ask)
        logger.info(f"VQA Question: {vqa_question}")
        
        backend_root = Path(__file__).resolve().parent.parent.parent
        keyframes_root = backend_root / "src" / "data" / "Keyframes"
        
        for doc in rerank_pool:
            vqa_score = 0.0
            frame_name = doc.get("frame_name")
            split = doc.get("split", "")
            
            # Since some documents might come from text search, they might miss "split".
            # Faiss search results usually don't have "split" unless merged from metadata.
            # In our system, the UI expects "image" or "folder_key" + "frame_name".
            # For local testing, we'll try to find the image on disk.
            
            # First try direct path if we know it
            img_path = None
            if frame_name:
                possible_path = keyframes_root / split / frame_name
                if possible_path.exists():
                    img_path = possible_path
                else:
                    possible_path = keyframes_root / frame_name
                    if possible_path.exists():
                        img_path = possible_path
            
            if img_path:
                vqa_score = reranker_service.score_image(str(img_path), vqa_question)
            
            # Combine scores: e.g. 70% original fusion score + 30% VQA confidence
            old_score = doc["final_score"]
            new_score = (old_score * 0.7) + (vqa_score * 0.3)
            
            doc["final_score"] = new_score
            doc["score_breakdown"]["vqa"] = vqa_score
            
        # Re-sort the reranked pool
        rerank_pool.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Combine back
        return rerank_pool + bottom_pool

    return initial_results


