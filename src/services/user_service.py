import sys
import os
import numpy as np
import random
import json
import logging

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.config.settings import get_settings

# Heavy legacy model factories are imported only by the endpoint that needs
# them. Keeping patchable placeholders also preserves test injection.
VLMProcessor = None
TRAKE = None
ElasticProcessor = None

settings = get_settings()

_trake_search = None
_vlm_processor = None
_elastic_processor = None

def get_trake_search():
    global _trake_search
    if _trake_search is None:
        factory = TRAKE
        if factory is None:
            from utils.trake_processing import TRAKE as factory
        _trake_search = factory()
    return _trake_search

def get_vlm_processor():
    global _vlm_processor
    if _vlm_processor is None:
        factory = VLMProcessor
        if factory is None:
            from utils.vlm_processing import VLMProcessor as factory
        _vlm_processor = factory()
    return _vlm_processor

def get_elastic_processor():
    global _elastic_processor
    if _elastic_processor is None:
        factory = ElasticProcessor
        if factory is None:
            from src.utils.elastic_processing import ElasticProcessor as factory
        _elastic_processor = factory()
    return _elastic_processor

def generate_random_answer():
    answers = [
        "This is an example answer",
        "Random response generated",
        "Here is your sample answer",
        "Auto-generated answer",
        "Dynamic answer text"
    ]
    return random.choice(answers)

# def getImageData():
#     # New logic using metadata.json (DictImagePath)
#     result = []
    
#     # Ensure we don't sample more items than exist
#     num_items_to_sample = min(150, len(DictImagePath))
#     random_img_ids = random.sample(list(DictImagePath.keys()), num_items_to_sample)

#     for id, img_id in enumerate(random_img_ids):
#         info = DictImagePath.get(img_id)
#         if not info:
#             continue

#         try:
#             # Re-use logic from getImageDataSingleTextSearch
#             filename_parts = info['frame_name'].split('.')[0].split('_')
#             folder_key = filename_parts[1]
#             video_key = filename_parts[2]
#             frame_key = filename_parts[3]

#             frame_number = info.get('global_frame_id')
#             fps = 25.0  # Default FPS
#             timestamp = frame_number / fps if frame_number is not None else None
            
#             full_image_path = os.path.join(SRC_DIR, 'data', 'New Keyframes', info['split'], info['frame_name'])
            
#             with open(full_image_path, "rb") as image_file:
#                 encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
#             result.append({
#                 'id': id,
#                 'folder_key': folder_key,
#                 'video_key': video_key,
#                 'frame_key': frame_key, 
#                 'timestamp': timestamp,
#                 'image': encoded_string,
#                 'answer': generate_random_answer()
#             })
#         except (FileNotFoundError, KeyError, IndexError):
#             # Skip if file is not found or info is malformed
#             continue

#     return result


def getImageDataSingleTextSearch(query, k):
    """Real visual text search: active backend's text encoder -> exact FAISS IP search.

    This is the production `/singletextsearch` path. The active backend is
    RETRIEVAL_BACKEND (BEiT3 or Jina CLIP v2, see
    src/services/retrieval_backend.py); scores are the real FAISS
    inner-product similarity, never a rank-derived placeholder.
    """
    from src.services.retrieval_backend import get_active_retriever

    return get_active_retriever().search_visual(query, top_k=k)


def getGroundedQASearch(query, k):
    """Return grounded Q&A frames and answer metadata."""
    text_query = query.strip()
    if not text_query:
        return [], {"status": "uncertain", "answer": "", "confidence": 0.0}
    from src.services.grounded_qa_service import grounded_video_qa

    return grounded_video_qa(text_query, top_k=k)


def getImageDataQAndASearch(query, k):
    """Compatibility wrapper returning only Q&A source frames."""
    frames, _summary = getGroundedQASearch(query, k)
    return frames

def getImageSearchById(image_id, k):
    """Search similar keyframes using the active backend's vector ID.

    The id is reconstructed from -- and searched against -- whichever FAISS
    index RETRIEVAL_BACKEND selects, so a Jina result row's vector id is
    always resolved in the Jina index and a BEiT3 one in the BEiT3 index;
    the two id spaces never cross.
    """
    from src.services.retrieval_backend import BackendPreparingError, get_active_retriever
    try:
        return get_active_retriever().search_by_vector_id(int(image_id), top_k=k)
    except BackendPreparingError:
        raise  # retryable 503, not an empty result set
    except Exception as e:
        logging.error(f"Error during image search by id: {e}")
        return []



def getCaptureSimilarSearch(image_path, k):
    """Similar-frame search seeded by a captured frame's exact preview still.

    Encodes the server-extracted still with the active backend's vision tower
    (BEiT3, or Jina CLIP v2 when RETRIEVAL_BACKEND=jina_clip_v2) and searches
    that backend's 1024-d FAISS index. The captured frame's per-video
    ``frame_idx`` is never passed as a global FAISS vector id, so this cannot
    silently return matches for an unrelated corpus frame. Errors propagate so
    the caller can report them instead of inventing results.
    """
    from src.services.retrieval_backend import get_active_retriever

    return get_active_retriever().search_by_image(image_path, top_k=k)


def getImageSearchByFile(image_file, k):
    """Search similar keyframes from an uploaded image via the active backend's
    vision tower.

    The uploaded image is encoded the same way that backend's indexed
    keyframes were and searched against its 1024-d FAISS index, so results
    carry real vector ids in that backend's own space (a follow-up pivot on
    them is well defined).
    """
    from src.services.retrieval_backend import BackendPreparingError, get_active_retriever

    try:
        return get_active_retriever().search_by_image(image_file, top_k=k)
    except BackendPreparingError:
        raise  # retryable 503, not an empty result set
    except Exception as e:
        logging.error(f"Error during image file search: {e}")
        return []

def GetImageDataTrakeSearch(query, top_results=100): 
    return get_trake_search().process_temporal_search(query, top_results=top_results)

def getTextSearchOCR(query: str, topk: int = 100):
    return get_elastic_processor().search_ocr(query, topk=topk)

def getTextSearchASR(query: str, topk: int = 100):
    return get_elastic_processor().search_asr(query, topk=topk)
