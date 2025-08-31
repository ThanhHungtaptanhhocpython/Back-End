from PIL import Image
from transformers import pipeline
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils.nlp_processing import Translation


class VLMProcessor:
    def __init__(self):
        self.vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
        self.translater = Translation()

    def load_and_resize(self, path, size=(384, 384)):
        img = Image.open(path).convert("RGB")
        return img.resize(size)

    def batch_answer(self, image_paths, question, batch_size=4):
        question = self.translater(question)
        # Load toàn bộ ảnh
        images = [self.load_and_resize(p) for p in image_paths]
        
        # Tạo danh sách input theo format yêu cầu
        inputs = [{'image': img, 'question': question} for img in images]
        
        # Chia thành các batch nhỏ
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_results = self.vqa_pipeline(batch)
            results.extend(batch_results)
        
        answer_list = [ans[0]['answer'] for ans in results]
        return answer_list
