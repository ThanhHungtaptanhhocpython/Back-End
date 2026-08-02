import os, glob, json, argparse
import torch
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import XLMRobertaTokenizer
from timm import create_model
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from sklearn.metrics.pairwise import cosine_similarity



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BEiT3:
    def __init__(self, model_checkpoint, tokenizer_checkpoint):
        # Load model
        checkpoint = torch.load(model_checkpoint, map_location=device)
        self.model = create_model('beit3_large_patch16_384_retrieval')  # bạn cần import create_model
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device).eval()

        # Load tokenizer
        self.tokenizer = XLMRobertaTokenizer(tokenizer_checkpoint)

        # Processor
        self.image_processor = transforms.Compose([
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
    
    def process_image(self, image):
        try:
            img = Image.open(image).convert("RGB")
        except:
            img = image  # nếu đã là PIL Image hoặc Tensor
        img = self.image_processor(img).unsqueeze(0).to(device)
        return img
    
    def process_text(self, text, max_len=64):
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - num_tokens)
        return torch.tensor([language_tokens], device=device), torch.tensor([padding_mask], device=device)
    
    def encode_image(self, image):
        img_tensor = self.process_image(image)
        with torch.no_grad():
            img_feat, _ = self.model(image=img_tensor, only_infer=True)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        return img_feat[0].cpu().numpy().astype(np.float32)
    
    def encode_text(self, text):
        text_tokens, padding_mask = self.process_text(text)
        with torch.no_grad():
            _, text_feat = self.model(text_description=text_tokens, padding_mask=padding_mask, only_infer=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat[0].cpu().numpy().astype(np.float32)
    
    def similarity(self, image, text):
        img_feat = self.encode_image(image)
        text_feat = self.encode_text(text)
        return cosine_similarity([img_feat], [text_feat])[0][0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_dataset(path):
    import kagglehub
    path = kagglehub.dataset_download(path)
    return path

def preprocess_data(keyframes_dir):
    all_keyframe_paths = dict()
    for video_dir in sorted(os.listdir(keyframes_dir)):
        data_path = os.path.join(keyframes_dir, video_dir)
        if not os.path.isdir(data_path):
            continue
        keyframe_paths = sorted(glob.glob(os.path.join(data_path, '*.webp')))
        all_keyframe_paths[video_dir] = keyframe_paths
    return all_keyframe_paths


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        choices=["extract", "init"],
        nargs="+",
        required=True,
        help="Choose one or more tasks"
    )
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_args()
    bs = args.batch_size
    keyframes_dir = download_dataset(args.ds_path)
    print(f"[Dataset] {keyframes_dir}")

    # --- Load BEiT3 model ---
    beit3 = BEiT3(args.model_checkpoint, args.tokenizer_checkpoint)

    all_keyframe_paths = preprocess_data(keyframes_dir)
    save_dir = './metaclip_features'
    os.makedirs(save_dir, exist_ok=True)

    # ========== Feature extraction ==========
    if "extract" in args.tasks:
        print("======== Start extracting BEiT3 features ========")
        for part, video_keyframe_paths in tqdm(all_keyframe_paths.items()):
            os.makedirs(os.path.join(save_dir, part), exist_ok=True)
            for video_id, keyframe_paths in tqdm(video_keyframe_paths.items(), leave=False):
                save_path = os.path.join(save_dir, part, f"{video_id}.npy")
                if os.path.exists(save_path):
                    print(f"[Skip] {save_path} already exists, skipping...")
                    continue

                video_feats = []
                for i in range(0, len(keyframe_paths), bs):
                    batch_paths = keyframe_paths[i:i+bs]
                    images = [beit3.process_image(p) for p in batch_paths]
                    images = torch.cat(images).to(device)

                    with torch.no_grad():
                        image_feats, _ = beit3.model(image=images, only_infer=True)
                    image_feats /= image_feats.norm(dim=-1, keepdim=True)

                    video_feats.extend(image_feats.cpu().numpy().astype(np.float32))

                    del images, image_feats
                    torch.cuda.empty_cache()

                np.save(save_path, np.array(video_feats, dtype=np.float32))

    # ========== Indexing ==========
    if "init" in args.tasks:
        print("======== Start indexing faiss ========")
        # Lấy dim feature từ model BEiT3
        feature_shape = beit3.model.visual.output_dim
        index = faiss.IndexIDMap(faiss.IndexFlatIP(feature_shape))
        metadata = {}

        if os.path.exists("metadata.json"):
            with open("metadata.json", "r") as f:
                metadata = json.load(f)

        for part in tqdm(sorted(os.listdir(save_dir))):
            part_dir = os.path.join(save_dir, part)
            for feature_path in tqdm(sorted(glob.glob(os.path.join(part_dir, '*.npy'))), leave=False):
                video_id = os.path.splitext(os.path.basename(feature_path))[0]
                csv_path = os.path.join(keyframes_dir, part, video_id, f"{video_id}_keyframes_metadata.csv")
                df = pd.read_csv(csv_path)
                feats = np.load(feature_path).astype(np.float32)
                feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8

                global_ids = df["global_frame_index"].astype(int).to_numpy()
                skip_all = all(str(gfid) in metadata for gfid in global_ids)
                if skip_all:
                    print(f"[Skip] {video_id} already indexed")
                    continue

                index.add_with_ids(feats, global_ids)
                for i, gfid in enumerate(global_ids):
                    metadata[str(gfid)] = {
                        "video_id": video_id,
                        "frame_name": df["keyframe_filename"].iloc[i],
                        "frame_index": int(i),
                        "split": part,
                    }

        faiss.write_index(index, os.path.join(save_dir, "faiss_index.bin"))
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
