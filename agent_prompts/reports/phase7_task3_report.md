# Phase 7 Task 3 Report: Qdrant Migration Evaluation

## Status: ✅ Completed

---

## What Changed

1. **Created `scripts/qdrant_migration_plan.py`**:
   - Designed a production-ready database schema for **Qdrant**.
   - Instead of maintaining 3 separate Faiss files on disk, Qdrant allows us to use **Named Vectors** (`openclip` 512-dim, `siglip` 768-dim, `beit3` 768-dim). 
   - A single image now acts as a container for all 3 vectors simultaneously.

2. **Metadata Payload Filtering**:
   - Defined the payload structure (`video_id`, `frame_name`, `timestamp`).
   - Faiss only supports returning integer IDs, forcing us to do heavy `JSON` lookups manually in Python. Qdrant stores the metadata (JSON) alongside the vectors and returns everything in a single query.

---

## Why These Decisions

- **Why Qdrant instead of Faiss?**:
  - **Memory Limits**: Faiss holds everything in RAM. Loading 3 giant embedding datasets could trigger an Out-Of-Memory (OOM) crash on the backend. Qdrant uses Memory-Mapped (mmap) files, meaning it smoothly swaps data between RAM and the SSD.
  - **Pre-filtering**: With Faiss, if a user wants to search *only* within video "V001", we have to search the entire database and filter it *afterwards* in Python. Qdrant allows us to apply a SQL-like `WHERE video_id = "V001"` filter *before* running the vector search, making it exponentially faster.
- **Why keep it Optional?**: Setting up Qdrant requires Docker and adds infrastructure overhead. For the upcoming AI challenge, the Faiss setup from Phases 1-6 is blazing fast and completely sufficient for a local environment. This migration script simply serves as an "Insurance Policy" in case the dataset grows too large.
