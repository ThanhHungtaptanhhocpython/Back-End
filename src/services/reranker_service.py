"""Reranker Service.

Provides Visual Question Answering (VQA) scoring for images using BLIP.
Extracts raw logits to return a probability of a 'yes' answer.
"""

import math
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class RerankerService:
    _instance = None
    _processor = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        """Lazy load the BLIP model to save memory."""
        if self._model is None:
            logger.info("Loading BLIP-VQA model (Salesforce/blip-vqa-base)...")
            try:
                from transformers import BlipProcessor, BlipForQuestionAnswering
                # Load model
                self._processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                self._model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
                logger.info("BLIP-VQA model loaded successfully.")
            except ImportError:
                logger.error("Failed to import transformers. Please install it.")
                raise

    def score_image(self, image_path: str, question: str) -> float:
        """
        Evaluate an image against a yes/no question using BLIP-VQA.
        
        Args:
            image_path: Absolute path to the image.
            question: The Yes/No question in English.
            
        Returns:
            A float between 0.0 and 1.0 representing the probability of "yes".
        """
        self._load_model()
        
        try:
            # 1. Load image
            raw_image = Image.open(image_path).convert('RGB')
            
            # 2. Process inputs
            inputs = self._processor(raw_image, question, return_tensors="pt")
            
            # 3. Generate 1 token with scores
            outputs = self._model.generate(
                **inputs, 
                max_new_tokens=1, 
                output_scores=True, 
                return_dict_in_generate=True
            )
            
            # 4. Extract logits for the first generated token
            logits = outputs.scores[0][0]  # shape: (vocab_size,)
            
            # 5. Get Token IDs for "yes" and "no"
            yes_id = self._processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_id = self._processor.tokenizer.encode("no", add_special_tokens=False)[0]
            
            yes_logit = logits[yes_id].item()
            no_logit = logits[no_id].item()
            
            # 6. Apply Softmax to get relative probability of "yes" vs "no"
            # formula: e^yes / (e^yes + e^no)
            
            # Stabilize softmax by subtracting the max
            max_logit = max(yes_logit, no_logit)
            exp_yes = math.exp(yes_logit - max_logit)
            exp_no = math.exp(no_logit - max_logit)
            
            prob_yes = exp_yes / (exp_yes + exp_no)
            return prob_yes
            
        except Exception as e:
            logger.error(f"Failed to score image {image_path}: {e}")
            return 0.0

# Singleton instance
reranker_service = RerankerService()
