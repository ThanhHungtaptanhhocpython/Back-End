# AI Challenge 2026 - Agentic AI Multimodal Retrieval System

> **Note for AI Agents:** For competition rules, query types (Textual KIS, Q&A, TRAKE), and scoring mechanics ($R@k$), see [agent_prompts/aic_2026_competition_context.md](agent_prompts/aic_2026_competition_context.md). For agent tasks and architecture, see `ARCHITECTURE_UPGRADE_PLAN.md` and `AGENT_TASKS.md`.

This repository contains the backend system for an advanced multimodal image and video keyframe retrieval system. Built with **FastAPI** and powered by **Agentic AI** (LangChain, Gemini/OpenAI), **Faiss**, **OpenCLIP**, and **Elasticsearch**, the system provides robust visual semantic retrieval, temporal sequence search, and a fully conversational reasoning AI assistant.

## 🚀 Key Features

- **Conversational KIS (Agentic AI):** An intelligent LangChain-powered agent that maintains session memory, uses the Spatiotemporal Reasoning (STAR) framework to select search tools, and synthesizes natural language answers using RAG.
- **Text-to-Image Search:** Retrieve video keyframes using natural language queries via CLIP embeddings.
- **Image-to-Image Search:** Find visually similar frames using a base image or Faiss ID.
- **Multimodal Search:** Query text overlays (OCR) and speech/dialogue (ASR) powered by Elasticsearch.
- **Temporal Search (TRAKE):** Retrieve sequences of events in chronological order with temporal constraint scoring.
- **Q&A Search (VQA):** Combines retrieval with a VLM (Vision-Language Model) processor to answer detailed questions about retrieved frames.
- **Self-Reflective Feedback Loop:** The Agent learns from user feedback (Thumbs Up/Down) and injects reflection context into its memory for subsequent queries.

## 📁 Directory Structure

```text
Back-End/
├── main.py                    # FastAPI entry point
├── config.py                  # Environment & Database Configurations
├── requirements.txt           # Python dependencies
├── scripts/                   # Data processing and indexing scripts (data_extraction, indexing, etc.)
├── src/                       
│   ├── agent/                 # Agentic AI Core
│   │   ├── llm_planner.py     # LangChain Agent Executor & STAR Prompt
│   │   ├── memory_manager.py  # Session-based Chat History
│   │   └── tools.py           # LangChain Tool wrappers (Vector, OCR, ASR, Temporal, VQA)
│   ├── api/                   
│   │   └── routers/           # FastAPI Routers (chat_router.py, search_router.py)
│   ├── schemas/               # Pydantic validation models
│   ├── services/              # Business logic (user_service.py, feedback_service.py)
│   ├── utils/                 # Core utilities (Faiss, VLM, ElasticProcessor)
│   ├── data/                  # Root directory for generated data (Keyframes, features)
│   └── dict/                  # Metadata and indices (faiss_index_clip.bin)
└── tests/                     # Unit tests
```

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    subgraph Conversational Agent Flow
        U[User Chat] --> V[FastAPI Chat Router]
        V --> W[LLM Planner Agent]
        W -- Tool Call: vector_search --> F[(Faiss / CLIP)]
        W -- Tool Call: ocr/asr_search --> E[(Elasticsearch)]
        W -- Tool Call: video_qa --> Q[BLIP-VQA Model]
        F --> W
        E --> W
        Q --> W
        W -- RAG Synthesis --> Z[Natural Language Response]
        FB[User Feedback] --> FS[Feedback Service]
        FS -- Reflection Note --> M[(Agent Memory)]
        M -. Context .-> W
    end
```

## ⚙️ Installation & Setup

### 1. Environment Setup (Windows/Linux)

It is highly recommended to use Conda to manage your environment to prevent dependency conflicts, especially with `faiss` and `torch`.

```bash
# Create and activate conda environment
conda create --name AIChallenge2025 python=3.10 -y
conda activate AIChallenge2025

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (.env)

Create a `.env` file in the root directory. For the Agentic AI features to work, you must provide an API Key:

```env
# Choose one of the following LLM providers:
GOOGLE_API_KEY="your_gemini_api_key"
OPENAI_API_KEY="your_openai_api_key"

# Other configurations...
```

### 3. Data Preparation

Before running the backend, you must build the local data assets. Ensure your source videos are placed in a reachable directory.

1. **Extract Keyframes:** `python scripts/data_extraction/rebuild_keyframes.py`
2. **Build Faiss Index:** `python scripts/indexing/build_clip_faiss_index.py`
3. **Index Elasticsearch:** Ensure Elasticsearch is running, then bulk insert OCR/ASR data.

#### Video playback / frame capture assets

The `/users/videos/{video_id}/playback` and `/users/videos/{video_id}/capture`
endpoints need two per-video assets:

| Asset | Env var | Default |
| --- | --- | --- |
| `media-info` (YouTube `watch_url` + `length`) | `MEDIA_INFO_PATH` | `media-info-aic25-b1.zip` at repo root, then `src/dict/media-info/` |
| `map-keyframes` (authoritative FPS) | `MAP_KEYFRAMES_PATH` | `src/dict/map-keyframes/`, then existing `src/dict/map-keyframes.zip` |

`media-info-aic25-b1.zip` is a large runtime asset and is **git-ignored** —
copy it onto the machine (or set `MEDIA_INFO_PATH` to wherever it lives). Both
values accept either a ZIP archive or an extracted directory, with or without a
wrapping top-level folder.

The YouTube timeline is assumed to be aligned with the dataset timeline, so the
playback offset defaults to `0` for every video. If a specific video is
verified to be shifted, add it to `PLAYBACK_OFFSETS_JSON`, e.g.
`PLAYBACK_OFFSETS_JSON={"L21_V029": -172}` (`source_time = playback_time - offset`).

## 🏃‍♂️ Running the Server

### Step 1: Start Elasticsearch
- **Via Docker:** `docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.15.0`

### Step 2: Start FastAPI
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*Note: The server uses port 8000 if you run via uvicorn with the `--port 8000` flag, or port 3000 by default if you just run `python main.py`.*

### Step 3: Test via Swagger UI
- Open your browser and navigate to: **http://localhost:8000/docs** (hoặc port 3000 tùy cách chạy)
- Use the **`/chat/conversational_kis`** endpoint to chat with the Agent.
- Use the **`/chat/feedback`** endpoint to send reflections to the Agent's memory.

## 🛣️ Agentic AI Upgrade Completed
The system has been successfully upgraded to an **Agentic AI Architecture**:
- ✅ **Phase 1:** Integrated LangChain Memory & Chat Interface.
- ✅ **Phase 2:** Refactored core search functions into LangChain `@tool` definitions.
- ✅ **Phase 3:** Implemented Spatiotemporal Reasoning (STAR) and RAG Synthesis via LLM Prompts.
- ✅ **Phase 4:** Established a Self-Reflective Feedback loop for continuous accuracy improvement.
- ✅ **Performance:** Wrapped agent tool executions in `asyncio.to_thread` to prevent FastAPI event loop blocking.
