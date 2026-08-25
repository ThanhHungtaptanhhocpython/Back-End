import sys
import os
import numpy as np
import random
import json
import base64
import logging
from PIL import Image

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.faiss_processing import MyFaiss
from utils.vlm_processing import VLMProcessor
from utils.trake_processing import TRAKE
from utils.elastic_processing import ElasticProcessor
from src.config.settings import get_settings

settings = get_settings()
bin_clip_file = str(settings.get_faiss_index_path())
meta_data = str(settings.get_metadata_path())

_cosine_faiss = None
_trake_search = None
_vlm_processor = None
_elastic_processor = None

def get_cosine_faiss():
    global _cosine_faiss
    if _cosine_faiss is None:
        _cosine_faiss = MyFaiss(bin_clip_file, meta_data)
    return _cosine_faiss

def get_trake_search():
    global _trake_search
    if _trake_search is None:
        _trake_search = TRAKE(get_cosine_faiss())
    return _trake_search

def get_vlm_processor():
    global _vlm_processor
    if _vlm_processor is None:
        _vlm_processor = VLMProcessor()
    return _vlm_processor

def get_elastic_processor():
    global _elastic_processor
    if _elastic_processor is None:
        _elastic_processor = ElasticProcessor()
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
    """Real visual text search: BEiT3 text encoder -> exact FAISS IP search.

    This is the production `/singletextsearch` path. It intentionally does
    NOT use the legacy OpenCLIP `MyFaiss` index (`get_cosine_faiss()`),
    which lives in a different embedding space than the BEiT3 corpus
    vectors. Returned scores are the real FAISS inner-product similarity,
    never a rank-derived placeholder.
    """
    from src.services.beit3_retriever import get_beit3_retriever

    return get_beit3_retriever().search_visual(query, top_k=k)


def getImageDataQAndASearch(query, k):
    """Retrieve candidate frames for Q&A prompts with the production BEiT3 index.

    The old Q&A path queried a legacy OpenCLIP FAISS index and then tried to
    load images from `src/data/Keyframes`. The current corpus is served through
    BEiT3 metadata/keyframe paths, so that legacy path could return zero frames
    even when the visual text search finds good candidates.
    """
    from src.services.beit3_retriever import get_beit3_retriever

    text_query = query.strip()
    if not text_query:
        return []

    results = get_beit3_retriever().search_visual(text_query, top_k=k)
    for item in results:
        item.setdefault('answer', '')
    return results

def getImageSearchById(image_id, k):
    """Search similar keyframes using BEiT-3 vector ID."""
    from src.services.beit3_retriever import get_beit3_retriever
    try:
        return get_beit3_retriever().search_by_vector_id(int(image_id), top_k=k)
    except Exception as e:
        logging.error(f"Error during BEiT-3 image search by id: {e}")
        return []



# res = getImageDataSingleTextSearch("Cycling race, overhead flycam angle: rider in blue-white overtakes three riders and leads until the finish.", 5)
# for item in res:
#     print(item['faiss_id_clip'])

# res2 = getImageSearchById(36244, 5)
# for item in res2:
#     print(item['frame_key'])
def getImageSearchByFile(image_file, k):
    result = []
    try:
        img = Image.open(image_file).convert("RGB")
        scores, image_ids, infos_query, image_paths = get_cosine_faiss().image_search_by_file(img, k)
        id = 0
        for info, image_path in zip(infos_query, image_paths):
            if not info:
                continue

            frame_name = info['global_frame_id']
            video_id = info['video_id']
            folder_key = info['split'].split('-')[1].upper()
            full_image_path = os.path.join(SRC_DIR, 'data', 'Keyframes', image_path)
            try:
                with open(full_image_path, "rb") as img_file_read:
                    encoded_string = base64.b64encode(img_file_read.read()).decode('utf-8')
            except Exception as e:
                logging.error(f"Failed to load image {full_image_path}: {e}")
                continue

            result.append({
                'id': id,
                'folder_key': folder_key,
                'video_key': video_id,
                'frame_key': frame_name,
                'image': encoded_string
            })
            id += 1 
        return result
    except Exception as e:
        logging.error(f"Error during image file search: {e}")
        return []

def GetImageDataTrakeSearch(query, top_results=100): 
    return get_trake_search().process_temporal_search(query, top_results=top_results)

def getTextSearchOCR(query: str, topk: int = 100):
    return get_elastic_processor().search_ocr(query, topk=topk)

def getTextSearchASR(query: str, topk: int = 100):
    return get_elastic_processor().search_asr(query, topk=topk)
