import os
import glob
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import glob
import json
import torch
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_dataset():
    import kagglehub
    # Download latest version
    path = kagglehub.dataset_download("trietdeptrai/keyframes-k01")
    return path

def prepocess_data(keyframes_dir):
    all_keyframe_paths = dict()
    for part in sorted(os.listdir(keyframes_dir)):
        data_part = part # L01, L02 for ex
        all_keyframe_paths[data_part] =  dict()

    for data_part in sorted(all_keyframe_paths.keys()):
        data_part_path = f'{keyframes_dir}/{data_part}'
        video_dirs = sorted(os.listdir(data_part_path))
        video_ids = [video_dir.split('_')[-1] for video_dir in video_dirs]
        for video_id, video_dir in zip(video_ids, video_dirs):
            keyframe_paths = sorted(glob.glob(f'{data_part_path}/{video_dir}/*.jpg'))
            all_keyframe_paths[data_part][video_id] = keyframe_paths
            
    return all_keyframe_paths

def get_OD_model():
    return easyocr.Reader(['vi'], gpu=True)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    bs = args.batch_size

    keyframes_dir = download_dataset()
    all_keyframe_paths = prepocess_data(keyframes_dir)
    reader = get_OD_model()
    save_dir = './OCR/feature'
    os.makedirs(save_dir, exist_ok=True)

    for key in tqdm(sorted(all_keyframe_paths.keys()), desc="Processing parts"):
        key_dir = os.path.join(save_dir, key)
        os.makedirs(key_dir, exist_ok=True)

        for video_id, frames in tqdm(all_keyframe_paths[key].items(), desc=f"Videos in {key}", leave=False):
            output_file = os.path.join(key_dir, f"{video_id}.json")
            if os.path.exists(output_file):
                print(f"[Skip] {output_file} đã có, bỏ qua...")
                continue

            video_ocr_results = {}
            for i in range(0, len(frames), bs):
                batch_paths = frames[i:i+bs]
                results = reader.readtext_batched(batch_paths, batch_size=bs)

                for img_path, result in zip(batch_paths, results):
                    refined = [item for item in result if item[2] > 0.5]
                    refined = easyocr.utils.get_paragraph(refined)
                    text_detected = [item[1] for item in refined]
                    video_ocr_results[os.path.basename(img_path)] = text_detected
                del results, batch_paths
                torch.cuda.empty_cache()

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(video_ocr_results, f, ensure_ascii=False, indent=2)

            print(f"[Saved] OCR results for {video_id} → {output_file}")
