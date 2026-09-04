# Roadmap KIS / TRAKE / VLM

## Checkpoint Execution Plan

Current constraint: the smooth keyframe rebuild path is paused because the new extraction is not dense enough in long scenes. The remaining work should proceed against the currently stable BEiT3/keyframe asset set, and every checkpoint must start with an asset consistency check.

### Checkpoint 1: Asset Baseline And Guard

Status: started.

Implemented:

- [x] Added `scripts/utils/check_asset_state.py`.
- [x] Checked current `KEYFRAMES_ROOT`, metadata, `map-keyframes`, and BEiT3 mapping.

Current findings:

- Current `KEYFRAMES_ROOT` scan has 904 video folders and 345,558 images; the active BEiT3 corpus references 286,629 images across 873 videos.
- `src/dict/metadata_clip.json` has 196,839 rows but points to the older `videos-l21-a/V001/keyframe_...` layout.
- Current BEiT3 `global_ids.parquet` sample paths match `KEYFRAMES_ROOT`.
- `metadata_clip.json` is intentionally retained with the legacy 196,839-vector `faiss_index.bin`; it must not be overwritten with BEiT3 rows.
- Added `src/dict/metadata_beit3.json`, generated in exact `global_ids.parquet` vector order with 286,629 rows and per-video FPS timestamps.
- Realigned all 12,845 ASR documents to `metadata_beit3.json` and recreated the `aic_asr` Elasticsearch index.

Command:

```powershell
python scripts\utils\check_asset_state.py
```

Pass condition:

- Metadata sample image paths exist under `KEYFRAMES_ROOT`.
- BEiT3 `global_ids.parquet` frame paths exist under `KEYFRAMES_ROOT`.
- BEiT3 FAISS `ntotal` equals `global_ids.parquet` rows when the machine has enough RAM to inspect the index.

### Checkpoint 2: Agent Search KIS Primary Query Execution

Status: implemented.

Implemented:

- [x] Added `agent_visual_query_limit` setting. Default is `1`.
- [x] Agent plans now expose `primary_visual_query`, `executed_visual_queries`, `support_visual_queries`, and `execution_strategy`.
- [x] Agent Search now executes only `executed_visual_queries` instead of every expanded visual query.
- [x] Support queries are still shown/kept for explanation, checklist context, and debugging.
- [x] Added regression tests for default primary-only execution and env override.

Why:

- The previous flow could dilute results by running several smaller support queries.
- KIS descriptions usually work better when the visual search receives one enriched holistic query first.
- Extra query variants should not be independent search branches by default.

Runtime tuning:

```env
AGENT_VISUAL_QUERY_LIMIT=1
```

Increase to `2` only when you intentionally want one support query to be searched too.

Verification:

```powershell
python -m pytest tests\test_agent_query_coordinator.py
```

### Checkpoint 3: Candidate Rerank And Evidence Layer

Status: implemented.

Implemented:

- [x] Light verifier now scores candidate-specific OCR/ASR evidence from `score_breakdown`, `ocr_text`, and `asr_text`.
- [x] Timestamp mapping is kept as a small evidence signal when `timestamp_source` is available.
- [x] Result metadata now includes `agent_verification.modality_evidence_score` and `agent_verification.score_breakdown`.
- [x] Direct retrieval evidence label now reflects the checkpoint 2 flow: `direct visual retrieval match`.
- [x] OpenRouter VLM verifier now selects candidates from a wider pool instead of blindly taking the first N frames.
- [x] VLM candidate selection now limits repeated frames from the same video via `agent_vlm_per_video_limit`.
- [x] VLM candidate IDs are assigned only after local image paths resolve, so missing images do not desync API result IDs.

Runtime tuning:

```env
AGENT_VLM_MAX_CANDIDATES=12
AGENT_VLM_CANDIDATE_POOL=40
AGENT_VLM_PER_VIDEO_LIMIT=3
```

Verification:

```powershell
python -m pytest tests\test_agent_query_coordinator.py tests\test_openrouter_vlm_verifier.py
```

### Checkpoint 4: VLM Reliability, Contract, And Cache

Status: implemented.

Implemented:

- [x] Added a strict structured-output JSON schema for OpenRouter VLM verdicts.
- [x] Rejects missing, duplicate, unexpected, or invalid candidate verdicts instead of using them for reranking.
- [x] Prompt requires visible evidence and explicit `matched` / `missing` constraints without invented details.
- [x] Added bounded retry with exponential backoff and retrieval fallback when the VLM request fails.
- [x] Added persistent verdict cache keyed by model, enriched query/checklist, frame identity, and image file version.
- [x] Cache is bounded by entry count and TTL and is stored under the ignored `.cache/` directory by default.
- [x] Verification summaries now expose `status`, `fallback_used`, cache hits/misses, API calls, retries, and contract errors.
- [x] Each verified frame exposes its VLM source (`api` or `cache`) and contract version.

Runtime tuning:

```env
AGENT_VLM_MAX_RETRIES=1
AGENT_VLM_RETRY_BACKOFF_SECONDS=0.5
AGENT_VLM_CACHE_ENABLED=true
AGENT_VLM_CACHE_PATH=
AGENT_VLM_CACHE_MAX_ENTRIES=5000
AGENT_VLM_CACHE_TTL_SECONDS=2592000
```

Verification:

```powershell
python -m pytest tests\test_agent_query_coordinator.py tests\test_openrouter_vlm_verifier.py
```

Next checkpoint:

### Checkpoint 5: TRAKE Ordered-Event Pipeline

Status: implemented.

Implemented:

- [x] TRAKE no longer initializes or queries the legacy 807 MB CLIP FAISS index.
- [x] Each event is translated/enriched independently and retrieved from the active BEiT3 corpus.
- [x] Candidate sequences are grouped by fully-qualified video ID and ordered by real timestamps.
- [x] Adjacent events and full sequences must fit configurable temporal windows.
- [x] Candidate pools are capped per event/video before beam search.
- [x] OCR/ASR evidence is matched by video and timestamp proximity before sequence ranking.
- [x] ASR evidence uses the BEiT3-aligned `aic_asr` index; legacy OCR IDs are not trusted for direct joins.
- [x] OpenRouter VLM verifies all event frames in a sequence together and returns matched/missing events.
- [x] VLM/API failure falls back to temporal-evidence ranking.
- [x] TRAKE responses retain the existing `frames`, `video_id`, and `timestamps` contract while exposing score, gaps, evidence, and VLM diagnostics.
- [x] Heavy legacy MyFaiss/BLIP/Elastic dependencies in `user_service` now load lazily.

Runtime tuning:

```env
TRAKE_RETRIEVAL_TOP_K=120
TRAKE_CANDIDATES_PER_EVENT_VIDEO=12
TRAKE_BEAM_WIDTH=40
TRAKE_MIN_EVENT_GAP_SECONDS=0
TRAKE_MAX_EVENT_GAP_SECONDS=300
TRAKE_MAX_SEQUENCE_SPAN_SECONDS=900
TRAKE_TEMPORAL_DECAY=0.01
TRAKE_EVIDENCE_WINDOW_SECONDS=12
TRAKE_OCR_ENABLED=true
TRAKE_ASR_ENABLED=true
TRAKE_VLM_ENABLED=true
TRAKE_VLM_MAX_SEQUENCES=5
```

Verification:

```powershell
python -m pytest tests\test_trake_checkpoint5.py tests\test_phase6_task1.py tests\test_phase6_task2.py tests\test_trake_video_grouping.py tests\test_agent_temporal_tool.py
```

Next checkpoint:

### Checkpoint 6: Grounded Q&A Pipeline

Status: implemented.

Implemented:

- [x] Q&A retrieves a BEiT3 visual candidate pool before generating an answer.
- [x] OCR/ASR evidence is retrieved for matching question intents and mapped to active BEiT3 frames.
- [x] Timestamp-based evidence mapping is rejected outside a configurable proximity window.
- [x] OpenRouter VLM receives only selected local keyframes and retrieved OCR/ASR snippets.
- [x] VLM output follows a strict answer/uncertain JSON contract with supporting frame IDs.
- [x] Low-confidence, invalid-contract, unavailable-image, and API-failure paths return `uncertain` instead of inventing an answer.
- [x] `/qnasearch` keeps the existing result-list contract and adds answer diagnostics under `data.meta`.
- [x] The frontend Q&A tab calls the grounded endpoint and displays only evaluated/supporting source frames.

Runtime tuning:

```env
QA_RETRIEVAL_POOL=40
QA_MAX_FRAMES=8
QA_PER_VIDEO_LIMIT=3
QA_TEXT_EVIDENCE_TOP_K=12
QA_EVIDENCE_WINDOW_SECONDS=15
QA_VLM_ENABLED=true
QA_MIN_CONFIDENCE=0.55
QA_MAX_TOKENS=700
```

Verification:

```powershell
python -m pytest tests\test_grounded_qa_service.py tests\test_task3.py tests\test_task4.py
```

Next checkpoint:

### Checkpoint 7: Jina Fine-Keyframe Runtime Integration

Status: implemented; production activation pending.

Implemented:

- [x] Added explicit `VISUAL_RETRIEVER=beit3|jina` selection with no silent
  cross-corpus fallback.
- [x] Added typed Jina model/device/revision/index/parquet settings.
- [x] Added a lazy singleton Jina CLIP v2 retriever using normalized
  1024-dimensional text/image embeddings and the final FAISS index.
- [x] Added startup validation for metric, dimension, count, contiguous IDs,
  duplicate keys, timestamps, and per-video totals.
- [x] Jina results use `source_frame_idx` for submission IDs and retain
  `keyframe_XXXX` only as keyframe identity.
- [x] Text Search, Agent Search, Q&A, TRAKE, timeline, vector-ID similarity,
  and image similarity use the selected retriever.
- [x] Added Jina cloud artifact names, manifest publisher, and
  `/health/retrieval?deep=true`.
- [x] Regression result: 99 passed, 1 skipped.

Activation still required:

- [ ] Publish `metadata/hcmai-assets-jina.json` using Azure credentials.
- [ ] Sync the four Jina artifacts into the backend cache.
- [ ] Pin the exact Hugging Face model revision used by image embedding.
- [ ] Switch `VISUAL_RETRIEVER=jina` and pass a real end-to-end query.
- [ ] Compare Jina and BEiT3 on the golden query set before removing rollback.

### Checkpoint 8: Remove BEiT3 After Jina Cutover

Status: blocked until Checkpoint 7 activation and evaluation pass.

Entry criteria:

- [ ] Jina passes `/health/retrieval?deep=true` with the real 693,124-vector corpus.
- [ ] KIS, Agent Search, Similar Search, Q&A, and TRAKE pass end-to-end tests on Jina.
- [ ] Jina meets or exceeds BEiT3 on the agreed golden-query retrieval metrics.
- [ ] The deployed Jina runtime completes a competition-style rehearsal without
  requiring BEiT3 rollback.
- [ ] Final Jina index, parquet metadata, model revision, and Azure manifest are
  backed up and reproducible.

Removal work:

- [ ] Make Jina the only visual retriever and remove `VISUAL_RETRIEVER=beit3`.
- [ ] Remove `src/services/beit3_retriever.py` and BEiT3-specific imports/tests.
- [ ] Move model-independent search, timeline, and result-mapping helpers out of
  the BEiT3 class before deleting it.
- [ ] Remove BEiT3 settings, Cloud Asset names, manifest entries, dependencies,
  documentation, and deployment variables.
- [ ] Remove or archive BEiT3 FAISS/metadata artifacts only after confirming no
  production process references them.
- [ ] Run the full backend regression suite and a fresh-deployment smoke test.

Exit criteria:

- [ ] Repository search finds no runtime BEiT3 references.
- [ ] Backend starts with only Jina artifacts and no BEiT3 environment variables.
- [ ] Search, Q&A, TRAKE, media mapping, and submission frame IDs remain correct.
- [ ] Rollback is provided by a tagged release/container and artifact backup,
  rather than BEiT3 code remaining in the active branch.

### Checkpoint 7A: Remap Legacy OCR Evidence To Jina

Status: implemented; waiting for the final Jina `global_ids.parquet` to run.

- [x] Added `scripts/data_extraction/new/remap_ocr_to_jina.py`.
- [x] Maps old OCR by `video_id + timestamp` to the nearest Jina keyframe.
- [x] Keeps Jina `vector_id`, `frame_path`, `source_frame_idx`, and alignment
  delta while preserving the original OCR text and legacy IDs for audit.
- [x] Rejects unmapped videos and suspicious timestamp matches beyond a
  configurable delta; it never overwrites the source OCR JSON.
- [ ] Run against `ocr_results (1).json` and the final Jina `global_ids.parquet`.
- [ ] Review alignment statistics and sample output before indexing `aic_ocr`.

Mục tiêu: cải thiện khả năng tìm đúng keyframe theo mô tả tự nhiên, đặc biệt cho KIS, TRAKE, Q&A và Agent Search.

## Trạng thái hiện tại

- [x] Đã sửa alignment ASR hiện tại: `src/dict/asr_results.json` không còn map phần lớn segment về `nearest_faiss_id = 0`.
- [x] Đã reindex Elasticsearch `aic_asr` từ ASR JSON đã sửa.
- [x] Đã thêm script rebuild alignment ASR: `scripts/data_extraction/new/realign_asr_keyframes.py`.
- [x] Đã sửa `extract_asr.py` để các lần chạy ASR sau align đúng theo `metadata_clip.json`.
- [x] Đã thêm script reindex ASR riêng: `scripts/indexing/reindex_asr.py`.

## Ưu tiên 1: Re-extract Keyframes Dày Hơn

Việc này ảnh hưởng lớn nhất đến độ chính xác KIS/TRAKE. Nếu keyframe bị thưa hoặc bỏ mất khoảnh khắc quan trọng, retrieval/rerank tốt cũng khó tìm đúng frame.

Cấu hình đề xuất:

```bash
--sampling-mode scene-uniform --sample-interval-seconds 2 --max-frames-per-scene 8
```

Checklist:

- [ ] Chạy notebook/keyframe extractor cho toàn bộ 800 video.
- [ ] Xuất keyframes mới.
- [ ] Xuất `map-keyframes` mới.
- [ ] Xuất metadata keyframe mới.
- [ ] Kiểm tra sample một vài video dài 20 phút xem keyframe có đủ mịn không.

GPU:

- Nên dùng GPU cho TransNetV2 và video processing nếu chạy số lượng lớn.
- Có thể chia 15 folder video cho nhiều account Kaggle chạy song song.

## Ưu tiên 2: Rebuild Visual Index

Sau khi keyframe thay đổi, FAISS/metadata visual phải rebuild lại. Nếu không, app vẫn tìm trên index cũ.

Checklist:

- [ ] Rebuild BEiT3/CLIP embeddings cho keyframes mới.
- [ ] Rebuild FAISS index.
- [ ] Rebuild `metadata_clip.json` đúng với FAISS id mới.
- [ ] Kiểm tra một số FAISS id có map đúng về frame/video/timestamp.

GPU:

- Nên dùng GPU. Đây là bước model inference trên rất nhiều ảnh.

## Ưu tiên 3: Realign OCR/ASR Theo Keyframes Mới

Vì vẫn là cùng 800 video, không cần chạy lại Whisper ASR nếu transcript đang ổn. Nhưng nếu keyframe/FAISS/metadata đổi, ASR phải align lại sang keyframe mới.

ASR flow:

```powershell
python scripts\data_extraction\new\realign_asr_keyframes.py --backup
python scripts\indexing\reindex_asr.py --recreate
```

Checklist:

- [ ] Realign ASR sau khi có `metadata_clip.json` mới.
- [ ] Reindex Elasticsearch `aic_asr`.
- [ ] Nếu OCR đang map theo keyframe cũ, rebuild/reindex OCR theo keyframes mới.

GPU:

- Realign ASR không cần GPU.
- Reindex Elasticsearch không cần GPU.
- OCR extraction nếu chạy lại có thể dùng CPU, nhưng GPU giúp nhanh hơn tùy OCR engine.

## Ưu tiên 4: Cải Thiện Agent Search KIS

Agent Search không nên tách mô tả thành quá nhiều query nhỏ làm mất ngữ cảnh. Nên làm giàu một query chính, giữ nguyên ràng buộc quan trọng.

Checklist:

- [ ] Prompt agent phải giữ nguyên chi tiết màu sắc, số lượng, thứ tự, vị trí, hành động.
- [ ] Agent tạo 1 query chính tiếng Anh giàu ngữ nghĩa, không tự bịa thuộc tính không có trong input.
- [ ] Agent có thể tạo thêm checklist xác minh, nhưng không dùng checklist như nhiều query độc lập mặc định.
- [ ] Search visual trước, OCR/ASR hỗ trợ khi mô tả có chữ/lời thoại/timestamp.
- [ ] Hiển thị rõ query tiếng Anh đã dùng để user kiểm tra.

GPU:

- Nếu dùng OpenRouter cho LLM/VLM: không cần GPU local.
- Nếu tự host VLM: cần GPU.

## Ưu tiên 5: VLM Rerank Cho KIS

Sau retrieval top candidates, dùng VLM để chọn frame giống mô tả nhất. Đây là bước giúp tăng độ chính xác khi mô tả nhiều chi tiết.

Flow đề xuất:

1. Agent làm giàu query.
2. Visual search lấy top 50-100 candidates.
3. Prefilter theo video/timestamp/OCR/ASR nếu có evidence.
4. Gửi top 20-50 frame cho VLM verifier.
5. VLM trả điểm match + lý do ngắn.
6. Reorder results theo VLM score.

Checklist:

- [ ] Cache VLM result theo `frame_id + prompt_hash`.
- [ ] Giới hạn số frame gửi VLM để tránh chậm và tốn API cost.
- [ ] Có fallback nếu VLM API fail: trả kết quả retrieval thường.
- [ ] UI hiển thị frame nào được VLM xác nhận cao nhất.

GPU:

- Dùng OpenRouter: không cần GPU local.
- Tự host VLM: cần GPU mạnh.

## Ưu tiên 6: Cải Thiện TRAKE

TRAKE cần hiểu chuỗi sự kiện theo thời gian, không chỉ search từng event rồi ghép theo `global_frame_id`.

Checklist:

- [ ] Parse mô tả thành các event có thứ tự.
- [ ] Với mỗi event, tạo enriched query nhưng vẫn giữ ngữ cảnh toàn cảnh.
- [ ] Retrieve candidates cho từng event.
- [ ] Ghép candidates theo cùng video hoặc video gần nhất.
- [ ] Dùng timestamp thật từ metadata, không chỉ `global_frame_id`.
- [ ] Áp temporal window: event sau phải xuất hiện sau event trước trong khoảng hợp lý.
- [ ] Dùng OCR/ASR evidence khi event có chữ, lời thoại, tên riêng, địa điểm, biển hiệu.
- [ ] Dùng VLM verifier để chấm sequence: toàn chuỗi có đúng mô tả không.

GPU:

- TRAKE logic không cần GPU.
- VLM verifier cần GPU nếu self-host, không cần GPU nếu dùng OpenRouter.

## Tối Ưu Tốc Độ TRAKE / VLM

Với setup dùng OpenRouter VLM, GPU local không làm VLM nhanh hơn vì inference chạy ở phía API. Muốn giảm latency thì nên tối ưu pipeline trước khi gọi VLM.

Checklist:

- [ ] Giảm số candidate đưa vào VLM rerank.
- [ ] Chỉ gọi VLM cho top 20-50 frame/candidate thay vì top 100.
- [ ] Dùng temporal window chặt hơn để loại sequence không hợp lý trước khi rerank.
- [ ] Dùng OCR/ASR/timestamp để prefilter trước khi gọi VLM.
- [ ] Cache kết quả VLM theo `frame_id + prompt_hash`.
- [ ] Có timeout/fallback khi OpenRouter chậm hoặc lỗi.
- [ ] Log số candidate trước/sau prefilter để biết bottleneck nằm ở retrieval, prefilter hay VLM.

GPU:

- Không giúp nhiều cho TRAKE logic thuần.
- Không giúp OpenRouter VLM nhanh hơn.
- Chỉ đáng kể nếu tự host VLM local hoặc rebuild embeddings/keyframes.
## Ưu tiên 7: Q&A Pipeline Thật

Hiện Q&A yếu vì chưa phải pipeline hỏi đáp hoàn chỉnh. Nó cần retrieve evidence rồi answer dựa trên frame/OCR/ASR.

Flow đề xuất:

1. Nhận câu hỏi.
2. Retrieve visual/OCR/ASR candidates.
3. Gom evidence theo video/time window.
4. Dùng VLM/LLM trả answer có căn cứ.
5. Trả kèm keyframes nguồn.

Checklist:

- [ ] Không chỉ trả frame rỗng answer.
- [ ] Có evidence text từ OCR/ASR.
- [ ] Có visual evidence từ frame.
- [ ] Answer phải chỉ dựa trên retrieved evidence.
- [ ] Nếu không đủ evidence, trả `không chắc` thay vì đoán.

GPU:

- Không cần GPU nếu dùng OpenRouter.
- Cần GPU nếu tự host VLM/LLM local.

## Thứ Tự Làm Khuyến Nghị

1. Re-extract keyframes dày hơn.
2. Rebuild visual embeddings + FAISS + `metadata_clip.json`.
3. Realign ASR theo metadata mới.
4. Reindex ASR/OCR Elasticsearch.
5. Tune Agent Search để làm giàu 1 query chính thay vì tách quá nhiều query nhỏ.
6. Thêm VLM rerank cho KIS.
7. Rebuild TRAKE theo temporal window + OCR/ASR + VLM verifier.
8. Sau cùng mới nâng Q&A thành pipeline answer thật.

## Ghi Chú Vận Hành

- Nếu chỉ sửa ASR alignment: không cần GPU.
- Nếu vẫn dùng cùng 800 video: không cần chạy lại Whisper, trừ khi transcript ASR sai nội dung.
- Nếu keyframe đổi: luôn realign ASR và reindex Elasticsearch.
- Nếu FAISS/metadata đổi: backend phải dùng đúng bộ index/metadata mới cùng phiên bản.
- Nếu dùng OpenRouter VLM: bottleneck là API latency, số frame gửi đi và rate limit, không phải GPU local.
