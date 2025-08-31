import sys
import os
import numpy as np
import random
import json
import base64

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, SRC_DIR)

from utils.faiss_processing import MyFaiss
from utils.combine_utils import merge_searching_results_by_addition

# json_path = os.path.join(SRC_DIR, 'dict', 'id2img_fps.json')
# bin_clip_file = os.path.join(SRC_DIR, 'dict', 'faiss_clip_cosine.bin')
# bin_clipv2_file = os.path.join(SRC_DIR, 'dict', 'faiss_clipv2_cosine.bin')
# fps_path = os.path.join(SRC_DIR, 'dict', 'fps.json')
metadata_path = os.path.join(SRC_DIR, 'dict', 'metadata_clip.json')
bin = os.path.join(SRC_DIR, 'dict', 'faiss_index_clip.bin')

CosineFaiss = MyFaiss(bin_clip_file, bin_clipv2_file, json_path)
DictImagePath = CosineFaiss.id2img_fps
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype('int64')

map_keyframes_path = os.path.join(SRC_DIR, "dict", "map_keyframes.json")
with open(map_keyframes_path, 'r') as f:
        KeyframesMapper = json.load(f)

# WARNING: Loading a 70MB JSON file into memory can cause performance issues.
with open(metadata_path, 'r') as f:
    Metadata = json.load(f)

with open(fps_path, 'r') as f:
    FpsDict = json.load(f)


def generate_random_answer():
    answers = [
        "This is an example answer",
        "Random response generated",
        "Here is your sample answer",
        "Auto-generated answer",
        "Dynamic answer text"
    ]
    return random.choice(answers)

def getImageData():

    random_video_ids = random.sample(list(KeyframesMapper), 3)
    result = []
    id = 0
    for key in random_video_ids:
        random_frame_ids = random.sample(list(KeyframesMapper[key]),150)
        folder_key, video_key = key.split('_', 1)
        for frame_key in random_frame_ids:
            
            image_path = f'./src/data/New Keyframes/{folder_key}/{video_key}/{frame_key.zfill(4) + ".webp"}'
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            result.append({
                    'id': id,
                    'folder_key': folder_key,
                    'video_key': video_key,
                    'frame_key': frame_key, 
                    'timestamp': KeyframesMapper[key][frame_key],
                    'image': encoded_string,
                    'answer': generate_random_answer()
                })
            id += 1

    return result


def getImageDataSingleTextSearch(clip, clipv2, text_query, k):
    result = []
    index = TotalIndexList
    k = min(k, len(index))

    if clip and clipv2:
        model_type = 'both'
    elif clip:
        model_type = 'clip'
    else:
        model_type = 'clipv2'
    
    if model_type == 'both':
        scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
        scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
        _, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2],
                                                          [list_clip_ids, list_clipv2_ids])
    else:
        _, list_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type=model_type)
    

    id = 0
    for img_id in list_ids:
        info = CosineFaiss.id2img_fps.get(img_id)
        if not info:
            continue

        
        image_path = info['image_path']
        parts = image_path.strip("/").split("/")   # ['data','KeyFrames','L02','V002','0128.webp']
        folder_key = parts[-3]                     # L02
        video_key = parts[-2]                      # V002
        frame_key = os.path.splitext(parts[-1])[0] # 0128

        
        frame_number = KeyframesMapper.get(f"{folder_key}_{video_key}", {}).get(str(int(frame_key)), None)
        video_id = f"{folder_key}_{video_key}"
        fps = FpsDict.get(video_id, 25.0)
        timestamp = frame_number / fps if frame_number is not None else None

        link = SceneId2Info.get(folder_key, {}).get(video_key, {}).get('video_metadata', {}).get('watch_url', '')
        
        full_image_path = os.path.join(SRC_DIR, 'data', 'Keyframes', folder_key, video_key, f"{frame_key}.webp")
        try:
            with open(full_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            continue

        result.append({
            'id': id,
            'folder_key': folder_key,
            'video_key': video_key,
            'frame_key': frame_key,
            'timestamp': timestamp,
            'image': encoded_string,
            'link': link
        })
        id += 1

    return result
