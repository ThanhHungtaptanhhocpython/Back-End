import random
import json
import base64

map_keyframes = "./src/dict/map_keyframes.json"

def getImageData():
    with open(map_keyframes, 'r') as f:
        KeyframesMapper = json.load(f)

    random_video_ids = random.sample(list(KeyframesMapper), 3)
    result = []
    id = 0
    for key in random_video_ids:
        random_frame_ids = random.sample(list(KeyframesMapper[key]),150)
        folder_key, video_key = key.split('_', 1)
        for frame_key in random_frame_ids:
            image_path = f'./src/data/Keyframes/{folder_key}/{video_key}/{frame_key.zfill(4) + ".webp"}'
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            result.append({
                    'id': id,
                    'folder_key': folder_key,
                    'video_key': video_key,
                    'frame_key': frame_key,
                    'timestamp': KeyframesMapper[key][frame_key],
                    'image': encoded_string
                })
            id += 1

    return result