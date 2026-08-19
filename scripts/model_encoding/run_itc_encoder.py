import os, glob, json, argparse
import torch
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from tqdm import tqdm
import open_clip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_dataset():
    import kagglehub
    path = kagglehub.dataset_download("trietdeptrai/keyframes-k01")
    return path

def prepocess_data(keyframes_dir):
    all_keyframe_paths = dict()
    for part in sorted(os.listdir(keyframes_dir)):
        data_part = part
        all_keyframe_paths[data_part] = {}
        data_part_path = f'{keyframes_dir}/{data_part}'
        video_dirs = sorted(os.listdir(data_part_path))
        video_ids = [video_dir.split('_')[-1] for video_dir in video_dirs]
        for video_id, video_dir in zip(video_ids, video_dirs):
            keyframe_paths = sorted(glob.glob(f'{data_part_path}/{video_dir}/*.webp'))
            all_keyframe_paths[data_part][video_id] = keyframe_paths
    return all_keyframe_paths

def get_CLIP_model():
    os.makedirs("clip_cache", exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14-quickgelu', pretrained='dfn5b',cache_dir ="./clip_cache"
    )
    model.eval()
    return model, preprocess

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_args()
    bs = args.batch_size
    keyframes_dir = download_dataset()
    print(f"[Dataset] {keyframes_dir}")

    model, preprocess = get_CLIP_model()
    model = model.to(device)

    all_keyframe_paths = prepocess_data(keyframes_dir)

    save_dir = './metaclip_features'
    os.makedirs(save_dir, exist_ok=True)

    # ========== Feature extraction ==========
    print("======== Start extract CLIP features ========")
    for key, video_keyframe_paths in tqdm(all_keyframe_paths.items()):
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)
        video_ids = sorted(video_keyframe_paths.keys())
        for video_id in tqdm(video_ids, leave=False):
            save_path = f'{save_dir}/{key}/{video_id}.npy'
            if os.path.exists(save_path):
                print(f"[Skip] {save_path} đã có, bỏ qua...")
                continue

            video_feats = []
            video_keyframe_path = video_keyframe_paths[video_id]
            for i in range(0, len(video_keyframe_path), bs):
                batch_paths = video_keyframe_path[i:i+bs]
                images = [preprocess(Image.open(p)).unsqueeze(0) for p in batch_paths]
                images = torch.cat(images).to(device)

                with torch.no_grad():
                    image_feats = model.encode_image(images)
                image_feats /= image_feats.norm(dim=-1, keepdim=True)

                for b in range(image_feats.shape[0]):
                    video_feats.append(image_feats[b].cpu().numpy().astype(np.float32))

                # cleanup
                del images, image_feats
                torch.cuda.empty_cache()

            np.save(save_path, np.array(video_feats, dtype=np.float32))

    # ========== Indexing ==========
    print("======== Start indexing faiss ========")
    feature_shape = model.visual.output_dim
    index = faiss.IndexIDMap(faiss.IndexFlatIP(feature_shape))
    metadata = {}

    # Nếu có metadata.json cũ thì load để skip
    if os.path.exists("metadata.json"):
        with open("metadata.json", "r") as f:
            metadata = json.load(f)

    for data_part in tqdm(sorted(os.listdir(save_dir))):
        part_dir = os.path.join(save_dir, data_part)
        for feature_path in tqdm(sorted(glob.glob(os.path.join(part_dir, '*.npy'))), leave=False):
            video_id = os.path.splitext(os.path.basename(feature_path))[0]
            csv_path = f"{keyframes_dir}/{data_part}/{video_id}/{video_id}_keyframes_metadata.csv"

            df = pd.read_csv(csv_path)
            feats = np.load(feature_path).astype(np.float32)

            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats = feats / norms
            global_ids = df["global_frame_index"].astype(int).to_numpy()

            # skip nếu global_id đã trong metadata
            skip_all = all(str(gfid) in metadata for gfid in global_ids)
            if skip_all:
                print(f"[Skip] {video_id} đã index rồi")
                continue

            index.add_with_ids(feats, global_ids)

            for i, gfid in enumerate(global_ids):
                metadata[str(gfid)] = {
                    "video_id": video_id,
                    "frame_name": df["keyframe_filename"].iloc[i],
                    "frame_index": int(i),
                    "split": data_part,
                }

    faiss.write_index(index, f"{save_dir}/faiss_index.bin")
    with open(f"{save_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
