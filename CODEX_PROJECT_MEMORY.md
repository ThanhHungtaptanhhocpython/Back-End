# AIC 2026 Project Memory And Competition Checklist

Last updated: 2026-09-02

This is the canonical handoff document for future Codex sessions. Read this file
before changing retrieval, cloud assets, metadata, deployment, Q&A, TRAKE, or
Agent Search. Do not infer that a generated artifact is active in the backend
until its runtime integration and an end-to-end search have both been verified.

## 1. Project Goal

Build a competition-ready multimodal video retrieval system for AIC 2026 with:

- Textual KIS: find the exact video/keyframe from a natural-language description.
- Q&A: localize the relevant scene and answer a grounded question.
- TRAKE: find one video and an ordered sequence of keyframes for multiple events.
- Agent Search: accept Vietnamese natural language, enrich it into one faithful
  English visual query, retrieve candidates, fuse OCR/ASR evidence, and rerank.

The scoring heavily rewards early precision. Correct results must be pushed into
the top 1/top 5, not merely appear somewhere in the top 100.

## 2. Current Architecture

```text
Browser frontend
    |
    v
FastAPI backend
    |-- visual query encoder -> FAISS search
    |-- Elasticsearch -> OCR/ASR retrieval
    |-- Agent coordinator -> query enrichment and modality routing
    |-- OpenRouter VLM -> top-candidate verification/reranking and grounded Q&A
    |-- Azure client -> private keyframe/video access
    |
    v
Azure Blob Storage (private)
    |-- keyframes/
    |-- embeddings/
    `-- metadata/
```

Deployment principle:

- Keep Azure Blob containers private.
- Keep FAISS and mapping artifacts on local disk/RAM of the backend after startup;
  do not query a FAISS index remotely from Blob Storage for every request.
- Return short-lived, read-only SAS URLs for images/videos to the frontend.
- Keep OpenRouter/Azure/Elasticsearch secrets in backend environment settings only.
- Do not use Colab as the production API host. It is acceptable for offline GPU
  extraction/embedding or as a temporary development service.

## 3. Confirmed Data And Artifacts

### Stable legacy/BEiT3 baseline

The existing runtime is currently BEiT3-based:

- Active BEiT3 corpus recorded in the roadmap: 286,629 vectors, 873 videos.
- `src/dict/metadata_beit3.json` follows BEiT3 `global_ids.parquet` vector order.
- `src/dict/metadata_clip.json` is legacy metadata and must stay paired with the
  legacy 196,839-vector FAISS index. Do not overwrite it with another corpus.
- ASR has been realigned to `metadata_beit3.json` and the `aic_asr`
  Elasticsearch index was recreated.
- Last reported ASR reindex result: 12,845 documents, 11,596 unique
  `nearest_faiss_id` values, zero documents missing `nearest_faiss_id`.

Baseline verification:

```powershell
python scripts\utils\check_asset_state.py
```

### New fine-keyframe Jina corpus

Fine keyframes were uploaded to the Azure `keyframes` container and embedded
with `jinaai/jina-clip-v2`, truncated to 1024 dimensions.

Confirmed final merge output from the notebook:

- Embedding run: `fine_keyframes_jina_clip_v2_1024d_v2`
- Model folder: `jina`
- Vector count: `693124`
- Video count: `873`
- Dimension: `1024`
- Metric: inner product over L2-normalized vectors
- Source: Azure per-video artifacts

Post-merge validation completed on 2026-09-02:

- FAISS/index/parquet integrity passed for all 693,124 vectors and 873 videos.
- `timestamp`, `source_fps`, and `source_frame_idx` coverage is 100%.
- All 873 videos start at `keyframe_0000`, confirming the current
  `map CSV n = keyframe file number + 1` convention.
- A stratified 20-frame sample passed Azure blob existence and map-keyframes
  timestamp/FPS/source-frame alignment checks.
- All four remote final artifacts matched local sizes and uploaded SHA-256
  metadata, and all 20 sampled JPEGs decoded/rendered successfully.
- Remaining manual check: seek several source videos to sampled timestamps and
  visually confirm the displayed video frame matches the keyframe.

Final runtime artifacts are in the Azure `embeddings` container under:

```text
indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/
    jina_faiss.index
    global_ids.parquet
    video_metadata.parquet
    index_meta.json
```

Per-video intermediate embeddings are under the same run's `records/` tree.
Use the `jina/` folder for backend search. Keep `records/` for recovery/remerge;
the backend should not search the individual record files.

Relevant notebooks:

- `scripts/notebooks/embed-jina-upload-azure-5jobs-disk-safe.ipynb`
- `scripts/notebooks/merge-azure-jina-embedding-index.ipynb`

Important: successful embedding and merge do not mean Jina is active in the
backend. As of this handoff, source/config searches show a BEiT3 retriever but no
production `JINA_*` retriever settings or Jina runtime service. Jina integration
is therefore the next P0 engineering task.

### Keyframe naming and timestamp mapping

Azure examples use paths such as:

```text
L22_a/L22_V001/keyframe_0001.jpg
```

The filename counter is a keyframe ordinal, not necessarily the original video
frame number or a timestamp. Timestamp/frame alignment must come from the
matching per-video map CSV/parquet fields (`pts_time`, `fps`, `frame_idx`) and
the final `global_ids.parquet`. Never treat a FAISS vector ID, keyframe ordinal,
source frame index, and timestamp as interchangeable.

Before activating Jina, sample at least 20 rows and verify:

- `frame_path` exists in Azure.
- Namespace and fully qualified `video_id` match the blob path.
- Timestamp maps to the expected moment in the source video.
- FAISS row `vector_id` maps to exactly the same parquet row.

## 4. Implemented Retrieval Checkpoints

Detailed implementation notes are in `ROADMAP_KIS_TRAKE_VLM.md`.

- [x] Checkpoint 1: BEiT3 asset baseline and metadata/ASR guard.
- [x] Checkpoint 2: Agent Search executes one enriched holistic visual query by
  default instead of diluting retrieval across many small queries.
- [x] Checkpoint 3: candidate evidence layer and diversified VLM candidate pool.
- [x] Checkpoint 4: strict VLM JSON contract, retries, fallback, and persistent cache.
- [x] Checkpoint 5: ordered-event TRAKE retrieval, temporal beam search, OCR/ASR
  evidence, and sequence-level VLM verification.
- [x] Checkpoint 6: grounded Q&A retrieval and VLM answer contract.

The above checkpoints were implemented against the BEiT3-aligned runtime. They
must be regression-tested after switching the visual retriever to Jina.

## 5. Agent Behavior Decisions

Agent Search should:

1. Accept Vietnamese descriptions directly.
2. Translate/enrich them into one holistic English visual query.
3. Preserve all stated constraints and avoid invented attributes. For example,
   do not add a color when the user did not specify one.
4. Keep support queries/checklists for explanation and reranking, but do not run
   every small clause as an independent visual search by default.
5. Retrieve broadly, prefilter with visual/OCR/ASR/timestamp evidence, then send
   only a small diverse candidate set to the VLM.
6. Fall back to retrieval ranking when OpenRouter is unavailable.

Recommended current VLM flow:

```text
Vietnamese query
  -> faithful enriched English query
  -> Jina visual retrieval (target runtime)
  -> OCR/ASR evidence fusion
  -> local candidate reduction/diversification
  -> OpenRouter VLM verification/rerank
  -> result grid with evidence and media URLs
```

Do not send all 100 frames to the VLM. Current checkpoint defaults select from a
wider pool and verify a limited, per-video-diverse set.

## 6. Azure Storage And Media Delivery

Storage account observed in the current setup:

```text
aicstorage2025data
```

Expected containers:

```text
keyframes
embeddings
metadata
```

The account/container should remain private. The earlier Azure Portal error when
generating SAS was an RBAC failure: the signed-in user selected `Account key`
but lacked `Microsoft.Storage/storageAccounts/listKeys/action`. Public access is
not required.

Production media flow:

```text
Frontend -> authenticated backend request
Backend  -> authorize user/result
Backend  -> create short-lived read-only User Delegation SAS
Frontend -> load image/video directly from Azure using that SAS URL
```

Preferred identity setup:

- Backend on Azure: use Managed Identity.
- Grant only required Blob data access.
- Grant the delegation action at storage-account/resource-group/subscription
  scope when generating User Delegation SAS.
- Use read-only SAS, HTTPS only, and a short expiry (for example 30-60 minutes).
- Never expose account keys, connection strings, `.env`, or OpenRouter keys to FE.

If backend hosting is outside Azure during development, a connection string can
be supplied as a backend secret, but it is not the preferred competition
deployment. SAS generation should be programmatic; do not manually generate a
token for every blob in Azure Portal.

Cloud asset support already exists in `docs/CLOUD_ASSETS.md`, but it is currently
documented and wired primarily around BEiT3 artifact names. It must be extended
or configured explicitly for the new Jina model/index files.

## 7. Runtime Services And Secrets

The local `.env` currently has values configured for BEiT3, Elasticsearch,
OpenRouter, Agent VLM, TRAKE, and Q&A. Values are intentionally not copied here.

Rules:

- `.env` remains local and must not be committed or uploaded to FE.
- Use the deployment platform's secret manager/environment variables.
- The frontend receives only public API configuration and expiring media URLs.
- Rotate any credential that was exposed in a notebook, screenshot, log, or Git.
- Remember that runtime settings may be stored in the per-user SQLite config
  store after first launch; `.env` edits alone may not become active. Use the
  Settings UI/API or set `HCMAI_DISABLE_CONFIG_STORE=1` for pure `.env` behavior.

## 8. Competition Deployment Target

Recommended minimum deployment:

```text
Static frontend host
    -> stable FastAPI backend host
       -> Jina text encoder
       -> CPU FAISS index in RAM
       -> Elasticsearch OCR/ASR
       -> OpenRouter VLM API
       -> private Azure Blob Storage
```

FAISS search does not require GPU. GPU mainly accelerates bulk image embedding
and query encoding. OpenRouter VLM inference runs remotely and does not consume
the project's GPU. A CPU backend may be acceptable after measuring Jina query
latency; otherwise isolate the Jina encoder on a stable GPU service. Do not rely
on a free Colab session during the competition.

## 9. Priority Checklist

### P0 - Activate and validate Jina search

- [x] Add explicit `JINA_MODEL_NAME_OR_PATH`, model/cache/device, truncate dimension,
  FAISS index, global IDs, video metadata, and index metadata settings.
- [x] Implement a Jina retriever that loads the model and final FAISS/parquet
  artifacts once, not per request.
- [x] Encode text with the same `jinaai/jina-clip-v2` model and 1024-dimensional
  truncation used for image embeddings.
- [ ] Pin/validate the exact model revision; dimension, normalization, metric, vector count,
  contiguous IDs, duplicate frame keys, and metadata paths at startup.
- [x] Make Jina selectable as the active visual retriever while retaining BEiT3
  as a rollback option until evaluation passes.
- [x] Connect Text Search, Agent Search, Q&A candidate retrieval, similar-frame
  search, and TRAKE event retrieval to the selected retriever abstraction.
- [x] Add unit and integration regression tests for selector, result mapping,
  Search, Agent, Q&A, TRAKE, timeline, and cloud paths (`99 passed, 1 skipped`).
- [ ] Run an end-to-end query against the real 693,124-vector Jina corpus.

### P0 - Publish Jina runtime assets safely

- [ ] Download/sync the four final `jina/` artifacts to backend local storage at
  startup and verify size/hash before activation.
- [ ] Add the Jina model snapshot or a pinned Hugging Face model revision to the
  deployment plan so startup is reproducible.
- [x] Extend cloud sync to describe the Jina artifacts and
  keyframe `.jpg` layout.
- [x] Add a separate `hcmai-assets-jina.json` publisher that reads existing Azure
  size/SHA-256 metadata without downloading the index.
- [ ] Publish `metadata/hcmai-assets-jina.json` (local Azure credential is not configured).
- [x] Keep a last-known-good local cache so a transient Azure outage does not
  prevent backend startup.

### P0 - Fix media delivery

- [ ] Add/verify backend generation of short-lived read-only User Delegation SAS
  URLs for keyframes and videos.
- [ ] Give backend identity only the required Azure RBAC roles.
- [ ] Ensure FE renders returned URLs and refreshes expired URLs via backend.
- [ ] Verify video seek uses authoritative timestamps and matches the keyframe.
- [ ] Configure CORS for the deployed frontend origin where required.

### P1 - Rebuild/realign text evidence for the final corpus

- [ ] Decide whether existing OCR text is complete enough for all 693,124 frames.
- [x] Add `scripts/data_extraction/new/remap_ocr_to_jina.py` to reuse legacy
  OCR by mapping `video_id + timestamp` to the nearest final Jina frame from
  `global_ids.parquet`. It writes a separate Jina-aligned JSON and never
  overwrites the OCR source file.
- [ ] Run the remapper with the more complete `ocr_results (1).json` source
  (164,786 records across 873 videos) and review its statistics before
  creating `aic_ocr`.
- [x] Add `scripts/notebooks/extract_ocr_azure_a100_batch32.ipynb` and its
  isolated `ocr_azure_worker.py` helper for full OCR of Jina keyframes directly
  from Azure. It uses A100 batch size 32, exact Jina metadata, per-video Azure
  checkpoints, and local per-video cache cleanup.
- [ ] Run one-video Azure OCR smoke test, then run the full corpus. Only invoke
  the notebook merge cell after every intended video checkpoint is complete.
- [ ] Map OCR and ASR evidence to Jina `vector_id`/frame paths using video ID and
  timestamp, never legacy FAISS IDs.
- [ ] Recreate Elasticsearch indices after alignment.
- [ ] Verify OCR-heavy Q&A, dialogue-heavy Q&A, and timestamp-sensitive TRAKE
  examples end to end.

ASR transcripts do not need Whisper re-extraction solely because visual
keyframes changed, provided the source videos are the same and transcripts are
correct. Their nearest-keyframe alignment and Elasticsearch documents do need
rebuilding for the final Jina corpus.

### P1 - Evaluate retrieval quality

- [ ] Build a golden set for KIS, Q&A, and TRAKE.
- [ ] Compare BEiT3 and Jina on Recall@1/5/20/50/100 and latency.
- [ ] Audit enriched queries for hallucinated colors, counts, actions, and order.
- [ ] Measure local retrieval, VLM rerank, and total request latency separately.
- [ ] Tune candidate pool and reranker size from measurements, not screenshots.
- [ ] Confirm visually that top results match descriptions and timestamps.

### P1 - Productionize backend

- [ ] Containerize or otherwise create a reproducible backend deployment.
- [ ] Provision enough RAM for the Jina FAISS index, metadata, model, and workers.
- [ ] Host Elasticsearch persistently or choose a managed equivalent.
- [ ] Add `/health` checks for Jina, FAISS, Elasticsearch, Azure, and OpenRouter.
- [ ] Add request timeouts, retries, rate limits, structured logs, and error IDs.
- [ ] Configure authentication if the competition deployment is not intentionally
  open to everyone.
- [ ] Deploy FE with the production backend base URL.

### P2 - Competition resilience

- [ ] Warm model/index/cache before the event.
- [ ] Cache query embeddings and VLM verdicts.
- [ ] Provide retrieval-only fallback when OpenRouter fails or rate-limits.
- [ ] Keep BEiT3 as a tested rollback until Jina wins the golden-set comparison.
- [ ] Back up final index/parquet/manifest and Elasticsearch snapshots.
- [ ] Run a full rehearsal from a fresh machine/container.
- [ ] Prepare a short operator runbook for restart, health checks, and rollback.

### P2 - Remove BEiT3 after Jina cutover (Checkpoint 8)

Do not start this checkpoint until the real Jina corpus passes health checks,
end-to-end workflows, golden-query evaluation, and a competition rehearsal.

- [ ] Promote Jina from selectable retriever to the only visual retriever.
- [ ] Extract any reusable search/timeline helpers currently inherited from the
  BEiT3 retriever into model-neutral code.
- [ ] Remove BEiT3 service code, settings, dependencies, tests, cloud artifact
  names, manifests, documentation, and deployment variables.
- [ ] Archive BEiT3 artifacts and preserve rollback through a tagged
  release/container plus verified artifact backup.
- [ ] Confirm a clean deployment starts and passes the full regression suite
  without BEiT3 files or environment variables.

## 10. Verification Commands

Current checkpoint regression suites:

```powershell
python -m pytest tests\test_agent_query_coordinator.py tests\test_openrouter_vlm_verifier.py
python -m pytest tests\test_trake_checkpoint5.py tests\test_phase6_task1.py tests\test_phase6_task2.py tests\test_trake_video_grouping.py tests\test_agent_temporal_tool.py
python -m pytest tests\test_grounded_qa_service.py tests\test_task3.py tests\test_task4.py
```

Run backend locally:

```powershell
python -m launcher --no-frontend
```

Manual alternative:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

Do not claim competition readiness until the final Jina end-to-end suite,
Azure SAS media flow, Elasticsearch health, and a golden-query rehearsal pass.

## 11. Instructions For The Next Codex Session

At the beginning of a new section/session:

1. Read this file and `ROADMAP_KIS_TRAKE_VLM.md`.
2. Run `git status --short`; do not revert unrelated user changes.
3. Verify actual code/artifacts before trusting a historical status statement.
4. Continue from the first unchecked P0 item unless the user changes priority.
5. Keep this document updated whenever an artifact path, vector count, active
   model, cloud layout, deployment decision, or checkpoint status changes.

Suggested user prompt for a new section:

```text
Read CODEX_PROJECT_MEMORY.md and continue from the first unchecked P0 item.
Verify the current repo state before editing.
```
