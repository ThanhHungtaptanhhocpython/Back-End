import os
import re
import json
import glob
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import transformers


transformers.logging.set_verbosity_error()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Image Preprocessing -------------------
def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # closest aspect ratio
    best_ratio = min(target_ratios, key=lambda r: abs((r[0]/r[1]) - aspect_ratio))
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    imgs = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    tensor_imgs = torch.stack([transform(img) for img in imgs])
    return tensor_imgs

# ------------------- Dataset -------------------
def download_dataset():
    import kagglehub
    path = kagglehub.dataset_download("trietdeptrai/frames2go")
    return path

def prepocess_data(keyframes_dir):
    all_keyframe_paths = dict()
    for part in sorted(os.listdir(keyframes_dir)):
        data_part_path = os.path.join(keyframes_dir, part)
        all_keyframe_paths[part] = {}
        for video_dir in sorted(os.listdir(data_part_path)):
            video_id = video_dir.split('_')[-1]
            keyframe_paths = sorted(glob.glob(f'{data_part_path}/{video_dir}/*.webp'))
            all_keyframe_paths[part][video_id] = keyframe_paths
    return all_keyframe_paths

# ------------------- OCR Model -------------------
def get_OCR_model():
    model_name = "5CD-AI/Vintern-1B-v3_5"
    try:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, use_flash_attn=False, cache_dir="./ocr_cache"
        ).eval().cuda()
    except:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, cache_dir="./ocr_cache"
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer

# ------------------- Arguments -------------------
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for OCR")
    args = parser.parse_args()
    return args

# ------------------- Main -------------------
if __name__ == "__main__":
    args = setup_args()
    bs = args.batch_size

    keyframes_dir = download_dataset()
    print("Dataset path:", keyframes_dir)

    all_keyframe_paths = prepocess_data(keyframes_dir)
    generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
    question = '<image>\nChỉ liệt kê tất cả chữ có trong ảnh, không mô tả, không thêm câu.'

    model, tokenizer = get_OCR_model()
    save_dir = './OCR/feature'
    os.makedirs(save_dir, exist_ok=True)

    for key in tqdm(sorted(all_keyframe_paths.keys())):
        key_dir = os.path.join(save_dir, key)
        os.makedirs(key_dir, exist_ok=True)

        for video_id in tqdm(sorted(all_keyframe_paths[key].keys())):
            output_file = os.path.join(key_dir, f"{video_id}.json")
            if os.path.exists(output_file):
                print(f"[Skip] {output_file} đã có, bỏ qua...")
                continue

            video_keyframe_path = all_keyframe_paths[key][video_id]
            video_results = {}

            for i in range(0, len(video_keyframe_path), bs):
                batch_paths = video_keyframe_path[i:i+bs]
                batch_tensors = [load_image(p, max_num=6) for p in batch_paths]
                batch_tensors = torch.cat(batch_tensors).to(torch.bfloat16).cuda()

                results = model.chat(tokenizer, batch_tensors, question, generation_config)
                del batch_tensors
                torch.cuda.empty_cache()

                if isinstance(results, str):
                    results = [results]

                for j, img_path in enumerate(batch_paths):
                    image_name = os.path.basename(img_path)
                    text_result = results[j] if j < len(results) else results[0]
                    lines = [l.strip() for l in text_result.split('\n') if l.strip()]

                    # fix long description
                    if len(lines) == 1 and len(lines[0].split()) > 45:
                        lines = re.findall(r'"(.*?)"', lines[0])

                    video_results[image_name] = list(dict.fromkeys(lines))  # unique

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(video_results, f, ensure_ascii=False, indent=2)
