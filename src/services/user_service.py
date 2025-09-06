import sys
import os
import numpy as np
import random
import json
import base64

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.faiss_processing import MyFaiss
from utils.vlm_processing import VLMProcessor
from utils.trake_processing import TRAKE

bin_clip_file = os.path.join(SRC_DIR, 'dict', 'faiss_index_clip.bin')
meta_data = os.path.join(SRC_DIR, 'dict', 'metadata_clip.json')

CosineFaiss = MyFaiss(bin_clip_file, meta_data)
TrakeSearch = TRAKE(bin_clip_file, meta_data)
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
    text_query = query.strip()
    result = []

    scores, list_ids, infos_query, image_paths = CosineFaiss.text_search(text_query, k)
    for info, image_path in zip(infos_query, image_paths):
        if not info:
            continue

        # Extract metadata
        frame_name = info['global_frame_id']
        video_id = info['video_id']
        # timestamp = info['pts']
        folder_key = info['split'].split('-')[1].upper()
        full_image_path = os.path.join(SRC_DIR, 'data', 'Keyframes', image_path)
        try:
            with open(full_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            continue

        id = 0
        result.append({
            'id': id,
            'folder_key': folder_key,
            'video_key': video_id,
            'frame_key': frame_name,
            # 'timestamp': timestamp,
            'image': encoded_string
        })
        id += 1

    return result


def getImageDataQAndASearch(query, k):
    text_query = query.strip()
    _, _, infos_query, image_paths = CosineFaiss.text_search(text_query, k)
    list_paths = []
    list_full_paths = []
    for info, image_path in zip(infos_query, image_paths):
        if not info:
            continue
        # Extract metadata
        frame_name = info['global_frame_id']
        video_id = info['video_id']
        # timestamp = info['pts']
        folder_key = info['split'].split('-')[1].upper()
        full_image_path = os.path.join(SRC_DIR, 'data', 'Keyframes', image_path)
        list_full_paths.append(full_image_path)
        try:
            with open(full_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            continue
        id = 0
        list_paths.append({
            'id': id,
            'folder_key': folder_key,
            'video_key': video_id,
            'frame_key': frame_name,
            # 'timestamp': timestamp,
            'image': encoded_string
        })
        id += 1
    vlm = VLMProcessor()
    answers = vlm.batch_answer(list_full_paths, text_query, batch_size=4)
   
    
    result = []
    for path, ans in zip(list_paths, answers):   
        result.append({
            'id': path['id'],
            'folder_key': path['folder_key'],
            'video_key': path['video_key'],
            'frame_key': path['frame_key'],
            'image': path['image'],
            'answer': ans,
            # thêm 'link': với link là đường dẫn đầy đủ tới video youtube
        })
        id += 1

    return result

def GetImageDataTrakeSearch(query): 
    return TrakeSearch.process_temporal_search(query)
