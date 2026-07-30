"""SigLIP Processing Wrapper.

Provides text feature extraction using the Google SigLIP model.
Model: google/siglip-base-patch16-224
Dimension: 768
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

class SigLIPProcessor:
    _instance = None
    _processor = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SigLIPProcessor, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        """Lazy load the SigLIP model to save memory during startup."""
        if self._model is None:
            logger.info("Loading SigLIP model (google/siglip-base-patch16-224)...")
            try:
                from transformers import AutoProcessor, AutoModel
                self._processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
                self._model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
                logger.info("SigLIP model loaded successfully.")
            except ImportError:
                logger.error("Failed to import transformers.")
                raise

    def get_text_features(self, text: str) -> np.ndarray:
        """
        Extract text embeddings using SigLIP.
        
        Args:
            text: Input text string.
            
        Returns:
            Numpy array of shape (768,) representing the text vector.
        """
        self._load_model()
        try:
            import torch
            inputs = self._processor(text=text, padding="max_length", return_tensors="pt")
            
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            
            # Normalize the vector (essential for Cosine Similarity)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to 1D numpy array
            return text_features.squeeze(0).cpu().numpy()
            
        except Exception as e:
            logger.error(f"SigLIP text extraction failed: {e}")
            # Fallback to zero vector of dimension 768
            return np.zeros(768, dtype=np.float32)

# Singleton instance
siglip_processor = SigLIPProcessor()
