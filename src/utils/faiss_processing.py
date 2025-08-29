import os
import clip
import open_clip
import torch
import json
import faiss
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils.nlp_processing import Translation

class MyFaiss:
    def __init__(self, bin_clip_file: str, bin_clipv2_file: str, json_path: str):    
        self.index_clip = self.load_bin_file(bin_clip_file)
        self.index_clipv2 = self.load_bin_file(bin_clipv2_file)

        self.id2img_fps = self.load_json_file(json_path)
        self.translater = Translation()
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/16", device=self.__device)
        self.clipv2_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', device=self.__device, pretrained='openai')
        self.clipv2_tokenizer = open_clip.get_tokenizer('ViT-B-32')
                                
    def load_json_file(self, json_path: str):
      with open(json_path, 'r') as f: 
        js = json.load(f)
      return {int(k):v for k,v in js.items()}
    
    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def text_search(self, text, index, k, model_type):
        text = self.translater(text)

        ###### TEXT FEATURES EXTRACTING ###### Embedding
        if model_type == 'clip':
            text = clip.tokenize([text]).to(self.__device)  
            text_features = self.clip_model.encode_text(text)
        else:
            text = self.clipv2_tokenizer([text]).to(self.__device)  
            text_features = self.clipv2_model.encode_text(text)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        if model_type == 'clip':
            index_choosed = self.index_clip
        else:
            index_choosed = self.index_clipv2
        
        if index is None:
          scores, idx_image = index_choosed.search(text_features, k=k)
        else:
          id_selector = faiss.IDSelectorArray(index)
          scores, idx_image = index_choosed.search(text_features, k=k, 
                                                   params=faiss.SearchParametersIVF(sel=id_selector))
        idx_image = idx_image.flatten()
        # print("idx_image: ")
        # print(idx_image)
        # print("list score: ")
        # print(scores)

        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        # print("image_paths: ")
        # print(image_paths)
        return scores.flatten(), idx_image, infos_query, image_paths

               