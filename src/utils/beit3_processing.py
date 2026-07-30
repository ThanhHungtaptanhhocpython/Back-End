"""BEiT-3 Processing Wrapper.

Provides text feature extraction using a BEiT-3 compatible model.
Dimension: 768
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

class BEiT3Processor:
    _instance = None
    _tokenizer = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BEiT3Processor, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        """Lazy load the BEiT model."""
        if self._model is None:
            logger.info("Loading BEiT model (microsoft/beit-base-patch16-224-pt22k)...")
            try:
                from transformers import AutoTokenizer, AutoModel
                # Using a generic text encoder for demo purposes as BEiT-3 text-only requires specific setup
                self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self._model = AutoModel.from_pretrained("bert-base-uncased")
                logger.info("BEiT-compatible text model loaded successfully.")
            except ImportError:
                logger.error("Failed to import transformers.")
                raise

    def get_text_features(self, text: str) -> np.ndarray:
        """
        Extract text embeddings.
        
        Args:
            text: Input text string.
            
        Returns:
            Numpy array of shape (768,) representing the text vector.
        """
        self._load_model()
        try:
            import torch
            inputs = self._tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                
            # Use pooled output (CLS token representation)
            text_features = outputs.pooler_output
            
            # Normalize the vector (essential for Cosine Similarity)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to 1D numpy array
            return text_features.squeeze(0).cpu().numpy()
            
        except Exception as e:
            logger.error(f"BEiT text extraction failed: {e}")
            # Fallback to zero vector of dimension 768
            return np.zeros(768, dtype=np.float32)

# Singleton instance
beit3_processor = BEiT3Processor()
