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
            # 'vector_id'; map them to the generic keys this module uses.
            faiss_list = getImageDataSingleTextSearch(visual_query, topk)
            for item in faiss_list:
                item["_score"] = item.get("score", 0.0)
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

    # 4. Merge and rank
    initial_results = merge_and_rank(visual_results, ocr_results, asr_results, weights)
    
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


