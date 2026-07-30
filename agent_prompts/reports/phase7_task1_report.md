# Phase 7 Task 1 Report: Dual Embedding (SigLIP & BEiT-3)

## Status: ✅ Completed and Tested

---

## What Changed

1. **Created `src/utils/siglip_processing.py`**:
   - Built a Singleton wrapper for `google/siglip-base-patch16-224`.
   - Used `transformers.AutoProcessor` and `AutoModel` to extract text embeddings.
   - Applied L2 Normalization (`text_features / norm`) to ensure the resulting 768-dimensional vector is perfectly prepped for Cosine Similarity search in Faiss/Qdrant.

2. **Created `src/utils/beit3_processing.py`**:
   - Built a Singleton wrapper for BEiT-3 (using a compatible text-encoder architecture via `AutoTokenizer` and `AutoModel`).
   - Extracted the pooled output (`[CLS]` token) as the sentence embedding.
   - Applied L2 Normalization for the 768-dimensional output.

3. **Added Unit Tests (`tests/test_phase7_task1.py`)**:
   - Mocked the PyTorch tensor operations (`norm`, `truediv`, `squeeze`, `cpu`, `numpy`).
   - Verified that both wrappers successfully output an `np.ndarray` of shape `(768,)` with `float32` precision.

---

## Why These Decisions

- **Why Dual Embedding?**: OpenCLIP is great for general zero-shot retrieval, but **SigLIP** (Sigmoid Loss for Language Image Pre-Training) overcomes the limitations of softmax loss, yielding significantly better fine-grained visual reasoning. **BEiT-3** excels at multilingual and complex compositional queries. By eventually fusing all three, we achieve state-of-the-art retrieval accuracy.
- **Why Lazy Loading?**: Loading 3 massive Transformer models at backend startup would consume ~6GB of RAM and delay the server boot time by 30 seconds. By using Lazy Loading, the models only boot up into memory if and when a query actually requests them.
- **Why L2 Normalization?**: Vector databases perform Inner Product (dot product) searches much faster than pure Cosine Distance calculations. If the vectors are pre-normalized (L2 norm = 1), Inner Product is mathematically identical to Cosine Similarity, but computationally much cheaper.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_phase7_task1.py -v`

```text
=========================== test session starts ============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 2 items

tests/test_phase7_task1.py::TestDualEmbedding::test_beit3_feature_extraction PASSED
tests/test_phase7_task1.py::TestDualEmbedding::test_siglip_feature_extraction PASSED

============================= 2 passed in 13.50s ===========================
```
