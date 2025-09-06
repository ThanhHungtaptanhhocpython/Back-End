import os
import json

import faiss
import numpy as np
import open_clip
import torch

from .nlp_processing import Translation

class MyFaiss:

    def __init__(self, bin_clip_file: str, json_path: str):
        """
        Initializes the MyFaiss instance.

        Args:
            bin_clip_file: Path to the binary file containing the Faiss index.
            json_path: Path to the JSON file containing metadata (id to image fps).
        """
        self.index_clip = self._load_bin_file(bin_clip_file)
        self.id2img_fps = self._load_json_file(json_path)
        self.translater = Translation()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'features')
        self.clip_model, self.clip_tokenizer = self._initialize_clip_model()

    def _initialize_clip_model(self) -> tuple[torch.nn.Module, any]:
        
        model_name = 'ViT-H-14-quickgelu'
        pretrained = 'dfn5b'
        
        # Load model onto CPU first to prevent memory errors on some systems
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, 
            device='cpu', 
            pretrained=pretrained
        )
        
        if self.device == "cuda":
            model = model.to(self.device)
            
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, tokenizer

    def _load_json_file(self, json_path: str) -> dict:
        """Loads a JSON file and converts its keys to integers."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}

    def _load_bin_file(self, bin_file: str) -> faiss.Index:
        """Loads a Faiss index from a binary file."""
        return faiss.read_index(bin_file)

    def _get_text_features(self, text: str) -> np.ndarray:
        """Translates and encodes the input text into a normalized feature vector."""
        translated_text = self.translater(text)
        tokens = self.clip_tokenizer([translated_text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy().astype(np.float32)

    def _search_faiss_index(self, text_features: np.ndarray, k: int, index_subset: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
        """Searches the Faiss index for the given text features."""
        if index_subset is None:
            scores, ids = self.index_clip.search(text_features, k=k)
        else:
            id_selector = faiss.IDSelectorArray(index_subset)
            params = faiss.SearchParametersIVF(sel=id_selector)
            scores, ids = self.index_clip.search(text_features, k=k, params=params)
        return scores.flatten(), ids.flatten()

    def _prepare_results(self, image_ids: np.ndarray) -> tuple[list[dict], list[str]]:
        """Maps image indices to metadata and constructs image paths."""
        infos_query = [self.id2img_fps.get(int(img_id)) for img_id in image_ids]
        # Filter out None results for IDs that were not found
        valid_infos = [info for info in infos_query if info]
        
        image_paths = [
            os.path.join(info['split'], info['video_id'], info['frame_name'])
            for info in valid_infos
        ]
        return valid_infos, image_paths

    def text_search(self, text: str, k: int, index: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
        
        text_features = self._get_text_features(text)
        scores, image_ids = self._search_faiss_index(text_features, k, index)
        infos_query, image_paths = self._prepare_results(image_ids)
        
        return scores, image_ids, infos_query, image_paths
    
    def _search_image_index(self, id: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        # Get feature vector for the given ID
        meta = self.id2img_fps.get(id)
        if not meta:
            raise ValueError(f"No feature found for ID {id}")
        
        video_id = meta["video_id"]
        split = meta["split"]
        frame_index = meta["frame_index"]
        
        feature_path = f"{self.save_dir}/{split}/{video_id}.npy"
        feats = np.load(feature_path).astype(np.float32)
        query_feature = feats[frame_index:frame_index+1]  # Reshape to (1, feature_dim)

        # Normalize query feature
        norms = np.linalg.norm(query_feature, axis=1, keepdims=True) + 1e-8
        query_feature = query_feature / norms

        # Perform Faiss search
        distances, indices = self.index_clip.search(query_feature, k)
        
        return distances[0], indices[0]
    
    def _prepare_results(self, image_ids: np.ndarray) -> tuple[list[dict], list[str]]:
        """Maps image indices to metadata and constructs image paths."""
        infos_query = [self.id2img_fps.get(int(img_id)) for img_id in image_ids]
        # Filter out None results for IDs that were not found
        valid_infos = [info for info in infos_query if info]
        
        image_paths = [
            os.path.join(info['split'], info['video_id'], info['frame_name'])
            for info in valid_infos
        ]
        return valid_infos, image_paths

    def image_search(self, id: int, k: int) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]: 
        scores, image_ids = self._search_image_index(id, k)
        infos_query, image_paths = self._prepare_results(image_ids)
        return scores, image_ids, infos_query, image_paths
