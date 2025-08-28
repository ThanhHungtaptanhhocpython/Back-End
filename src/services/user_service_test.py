import sys
import os
import numpy as np
import random
import json
import base64

# Add the 'src' directory to the Python path to resolve module import errors.
# This allows the script to be run from any directory.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, SRC_DIR)

from utils.faiss_processing import MyFaiss
from utils.combine_utils import merge_searching_results_by_addition

# Construct absolute paths for data files by joining the source directory
# with the relative file paths. This makes file access independent of the
# current working directory.
json_path = os.path.join(SRC_DIR, 'dict', 'id2img_fps.json')
bin_clip_file = os.path.join(SRC_DIR, 'dict', 'faiss_clip_cosine.bin')
bin_clipv2_file = os.path.join(SRC_DIR, 'dict', 'faiss_clipv2_cosine.bin')

CosineFaiss = MyFaiss(bin_clip_file, bin_clipv2_file, json_path)
DictImagePath = CosineFaiss.id2img_fps
TotalIndexList = np.array(list(range(len(DictImagePath)))).astype('int64')

# def getImageDataSingleTextSearch( clip, clipv2, text_query , k ):
#     result = []
#     index = TotalIndexList
#     k = min(k, len(index))

#     if clip and clipv2:
#       model_type = 'both'
#     elif clip:
#        model_type = 'clip'
#     else:
#        model_type = 'clipv2'

#     if model_type == 'both':
#       scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
#       scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
#       _, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2],
#                                                                   [list_clip_ids, list_clipv2_ids])
#       infos_query = list(map(CosineFaiss.id2img_fps.get, list(list_ids)))
#       list_image_paths = [info['image_path'] for info in infos_query]
#     else:
#       _, _, _, list_image_paths = CosineFaiss.text_search(text_query, index=index, k=k, model_type=model_type)

map_keyframes_path = os.path.join(SRC_DIR, "dict", "map_keyframes.json")

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
#     with open(map_keyframes, 'r') as f:
#         KeyframesMapper = json.load(f)

#     random_video_ids = random.sample(list(KeyframesMapper), 3)
#     result = []
#     id = 0
#     for key in random_video_ids:
#         random_frame_ids = random.sample(list(KeyframesMapper[key]),150)
#         folder_key, video_key = key.split('_', 1)
#         for frame_key in random_frame_ids:
            
#             image_path = f'./src/data/Keyframes/{folder_key}/{video_key}/{frame_key.zfill(4) + ".webp"}'
#             with open(image_path, "rb") as image_file:
#                 encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#             result.append({
#                     'id': id,
#                     'folder_key': folder_key,
#                     'video_key': video_key,
#                     'frame_key': frame_key, 
#                     'timestamp': KeyframesMapper[key][frame_key],
#                     'image': encoded_string,
#                     'answer': generate_random_answer()
#                 })
#             id += 1

#     return result


# def getImageDataSingleTextSearch(): 
#     with open(map_keyframes, 'r') as f:
#         KeyframesMapper = json.load(f)

#     random_video_ids = random.sample(list(KeyframesMapper), 3)
#     result = []
#     id = 0
#     for key in random_video_ids:
#         random_frame_ids = random.sample(list(KeyframesMapper[key]),150)
#         folder_key, video_key = key.split('_', 1)
#         for frame_key in random_frame_ids:
            
#             image_path = f'./src/data/Keyframes/{folder_key}/{video_key}/{frame_key.zfill(4) + ".webp"}'
#             with open(image_path, "rb") as image_file:
#                 encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#             result.append({
#                     'id': id,
#                     'folder_key': folder_key,
#                     'video_key': video_key,
#                     'frame_key': frame_key,
#                     'timestamp': KeyframesMapper[key][frame_key],
#                     'image': encoded_string,
#                 })
#             id += 1

#     return result

# def getImageDataSingleTextSearch(clip, clipv2, text_query, k, map_keyframes_path):
#     with open(map_keyframes_path, 'r') as f:
#         KeyframesMapper = json.load(f)

#     result = []
#     index = TotalIndexList
#     k = min(k, len(index))

#     # Xác định model nào sẽ dùng
#     if clip and clipv2:
#         model_type = 'both'
#     elif clip:
#         model_type = 'clip'
#     else:
#         model_type = 'clipv2'

#     # Truy vấn bằng CLIP / FAISS
#     if model_type == 'both':
#         scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
#         scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
#         _, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2],
#                                                           [list_clip_ids, list_clipv2_ids])
#     else:
#         _, list_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type=model_type)

#     # Duyệt qua kết quả tìm kiếm để lấy ảnh + encode base64
#     id = 0
#     for img_id in list_ids:
#         info = CosineFaiss.id2img_fps.get(img_id)
#         print(info)

#         if not info:
#             continue

#         folder_key, video_key = info['video_id'].split('_', 1)
#         frame_key = str(info['frame_id'])
#         timestamp = info.get('timestamp', KeyframesMapper.get(info['video_id'], {}).get(frame_key, None))

#         image_path = os.path.join(SRC_DIR, 'data', 'Keyframes', folder_key, video_key, f'{frame_key.zfill(4)}.webp')
#         try:
#             with open(image_path, "rb") as image_file:
#                 encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#         except FileNotFoundError:
#             continue

#         result.append({
#             'id': id,
#             'folder_key': folder_key,
#             'video_key': video_key,
#             'frame_key': frame_key,
#             'timestamp': timestamp,
#             'image': encoded_string,
#         })
#         id += 1

#     return result

# result = getImageDataSingleTextSearch(clip = True, clipv2 = True, text_query = "một người đàn ông đang lái xe", k = 5, map_keyframes_path = map_keyframes_path)
# print(result)
def getImageDataSingleTextSearch(clip, clipv2, text_query, k, map_keyframes_path):
    with open(map_keyframes_path, 'r') as f:
        KeyframesMapper = json.load(f)

    result = []
    index = TotalIndexList
    k = min(k, len(index))

    # Xác định model nào sẽ dùng
    if clip and clipv2:
        model_type = 'both'
    elif clip:
        model_type = 'clip'
    else:
        model_type = 'clipv2'

    # Truy vấn bằng CLIP / FAISS
    if model_type == 'both':
        scores_clip, list_clip_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clip')
        scores_clipv2, list_clipv2_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type='clipv2')
        _, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2],
                                                          [list_clip_ids, list_clipv2_ids])
    else:
        _, list_ids, _, _ = CosineFaiss.text_search(text_query, index=index, k=k, model_type=model_type)

    # Duyệt qua kết quả tìm kiếm để lấy ảnh + encode base64
    id = 0
    for img_id in list_ids:
        info = CosineFaiss.id2img_fps.get(img_id)
        # print(info)

        if not info:
            continue

        # Parse từ image_path thay vì video_id
        image_path = info['image_path']
        parts = image_path.strip("/").split("/")   # ['data','KeyFrames','L02','V002','0128.webp']
        folder_key = parts[-3]                     # L02
        video_key = parts[-2]                      # V002
        frame_key = os.path.splitext(parts[-1])[0] # 0128

        # Lấy timestamp (nếu có)
        timestamp = info.get('timestamp', KeyframesMapper.get(f"{folder_key}_{video_key}", {}).get(frame_key, None))

        # Mở ảnh
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
        })
        id += 1

    return result


# Test thử
result = getImageDataSingleTextSearch(
    clip=True,
    clipv2=True,
    text_query="một người đàn ông đang lái xe",
    k=5,
    map_keyframes_path=map_keyframes_path
)
print(result)
