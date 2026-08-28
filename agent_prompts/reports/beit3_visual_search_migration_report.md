# BEiT3 Visual Search Migration Report

## Status: ✅ Completed, tested with real artifacts, verified through the UI

Branch: `feature/beit3-visual-search` (off `main`), commit `0137efb`, pushed to `origin`.

---

## Summary

`POST /users/singletextsearch` used to run visual retrieval through an
OpenCLIP (`ViT-H-14-quickgelu` / `dfn5b`) FAISS index — an embedding space
that does not match the 1024-d BEiT3 corpus vectors already built for this
challenge (873 videos, 286,629 keyframes, scope L21–L30). It also had a fake
`src/utils/beit3_processing.py` that claimed to be BEiT3 but actually ran
`bert-base-uncased`, and a multimodal fusion path that faked visual scores
from rank position (`topk - i`) instead of real similarity.

This migration replaces that path end to end:

```
text query → real SentencePiece tokenizer → real BEiT3-large text encoder
→ normalized (1, 1024) query vector → faiss.IndexIDMap2(IndexFlatIP(1024))
exact search → global_ids.parquet lookup → structured API result
```

---

## What Changed

### 1. New: `src/utils/beit3_backbone.py`
A trimmed, inference-only `BEiT3ForRetrieval` architecture, adapted from
`microsoft/unilm/beit3` (`modeling_finetune.py` + `modeling_utils.py`, MIT
license). Keeps only the language/vision heads needed for inference; drops
the training-only `ClipLoss` criterion (verified to hold zero parameters, so
dropping it doesn't change the checkpoint's `state_dict` keys) and the
`timm` model-registry dependency — the class is constructed directly instead
of via `timm.create_model`. State-dict key names (`beit3.*`,
`language_head.*`, `vision_head.*`, `logit_scale`) are unchanged from
upstream, so the real `beit3_large_patch16_384_f30k_retrieval.pth` checkpoint
loads with **0 missing / 0 unexpected keys**.

### 2. New: `src/services/beit3_retriever.py`
The dedicated retrieval service (`BEiT3Retriever` class + lazy singleton
`get_beit3_retriever()`), matching the existing lazy-singleton convention
already used by `get_cosine_faiss()` in `user_service.py`. Owns:

- BEiT3 model loading + checkpoint validation
- SentencePiece tokenizer loading, using the historical fairseq/XLM-R id
  offset mapping (`<s>=0, <pad>=1, </s>=2, <unk>=3`, spm id 0 → unk,
  everything else shifted `+1`) — not a modern HF tokenizer
- FAISS index loading (`faiss.read_index`, dimension-checked against 1024)
- `global_ids.parquet` / `video_metadata.parquet` / `index_meta.json` loading
- **Schema-flexible column detection**: since the real parquet schema wasn't
  known in advance, columns are auto-detected against a candidate-name list
  (e.g. `vector_id`/`global_id`/`faiss_id`/`id`) with optional
  `BEIT3_COL_*` env overrides, and it fails loudly (listing the actual
  columns found) if the essential id/video columns can't be identified —
  rather than guessing a fixed schema.
- `encode_text(query) -> np.ndarray[1,1024]` with hard invariant checks
  (shape, finiteness, L2 norm ≈ 1)
- `search_visual(query, top_k) -> list[dict]` returning **real FAISS
  inner-product scores**, never rank-derived placeholders
- `_to_json_safe()`: converts pandas/numpy scalar values (NaN floats,
  `np.int64`, etc.) to native JSON-safe Python types before they reach the
  response — see "Bug found and fixed" below.

Every failure mode (missing checkpoint/index/tokenizer/parquet path, wrong
FAISS dimension, `ntotal` ≠ metadata row count, non-finite/wrong-shape/
unnormalized query vector, invalid `top_k`, bad `BEIT3_DEVICE`) raises
`BEiT3RetrieverError` loudly — no silent fallback to another encoder.

### 3. Modified: `src/services/user_service.py`
`getImageDataSingleTextSearch(query, k)` now delegates to
`get_beit3_retriever().search_visual(query, top_k=k)` instead of
`get_cosine_faiss().text_search(...)`. This is the only change to the
production `/singletextsearch` code path; `/imagesearch`, `/qnasearch`, and
`/trakesearch` still use the OpenCLIP `MyFaiss` index (out of scope — they
search a different, already-built OpenCLIP corpus).

### 4. Modified: `src/services/fusion_service.py`
Removed the fake score:

```python
# before
item["_score"] = float(topk - i)  # Descending fake score for normalization
item["faiss_id"] = item.get("faiss_id_clip")

# after
item["_score"] = item.get("score", 0.0)      # real FAISS IP score
item["faiss_id"] = item.get("vector_id")
```

### 5. Modified: `src/config/settings.py`
Added `BEIT3_*` Pydantic settings fields: `beit3_faiss_index_path`,
`beit3_global_ids_path`, `beit3_video_metadata_path`, `beit3_index_meta_path`,
`beit3_checkpoint_path`, `beit3_tokenizer_path`, `beit3_device`,
`beit3_max_seq_len`, plus optional `beit3_col_*` overrides. All default to
`None`/sensible values — no in-repo default path, since these are
machine-specific runtime artifacts that must never be committed.

### 6. Modified: `.env.example`, `requirements.txt`, `.gitignore`
- `.env.example` documents all new `BEIT3_*` variables.
- `requirements.txt` gained `pyarrow`, `sentencepiece`, `torchscale` (no
  `timm` needed — see backbone note above).
- `.gitignore` now ignores `*.pth`, `*.index`, `*.parquet`, `*.spm`, and
  `src/dict/nw/*.bin`.

### 7. New: `tests/test_beit3_retriever.py`
13 unit tests, runnable without the real checkpoint/index (which don't
exist in this dev environment — only on the deployment machine):
tokenizer offset-mapping against a real trained toy SentencePiece model,
parquet column auto-detection (including the "fails loudly" case), a
synthetic `faiss.IndexIDMap2(IndexFlatIP)` search + metadata-lookup
integration test, and query-vector invariant validation. All 13 pass.

---

## Bug found and fixed (via real-data testing)

Once the real artifacts were available, loading the real `global_ids.parquet`
showed its `timestamp_s` column is 100% `NaN` (0/286,629 rows populated).
Pandas `NaN` is a `float`, not `None` — left as-is, it would have serialized
into technically-invalid `NaN` JSON tokens instead of `null`, and `to_dict()`
also leaves numpy scalar dtypes (`np.int64`, etc.) in row values, which can
fail JSON serialization outright. Added `BEiT3Retriever._to_json_safe()` to
sanitize every field before it reaches the API response. Re-ran the 13 unit
tests after the fix — still pass.

---

## Branch history note

Work was originally done on `refactor/agentic-ai-architecture`. That branch
turned out to already be fully merged into `main` (PR #1, commit `a8210f9`),
with `main` two merges ahead of it. Every file touched by this change was
byte-identical between the two branch heads, so `git checkout main` carried
the uncommitted work over with zero conflicts. `feature/beit3-visual-search`
was branched from `main` and the work committed there; `main` itself was
never modified directly.

---

## Validation — real artifacts, not synthetic

The real runtime artifacts (`beit3_faiss.index`, `global_ids.parquet`,
`video_metadata.parquet`, `index_meta.json`,
`beit3_large_patch16_384_f30k_retrieval.pth`, `beit3.spm`) were provided and
moved to `E:\GitHub\HCMAI2026\beit3_runtime\` (outside the git working tree,
zero risk of accidental commit — confirmed via `git status`). One file
arrived as `beit3_large_patch16_384_f30k_retrieval.zip`; this turned out to
*be* the checkpoint already — PyTorch's own `.pth` serialization format is a
zip container internally (`archive/data.pkl`, `archive/data/<tensor-id>`,
`archive/version`) — so it was simply renamed to `.pth`, no re-encoding
needed. `.env` (gitignored) at the repo root points `BEIT3_*_PATH` at these
files, `BEIT3_DEVICE=cpu` (no GPU on the dev machine).

**Index / mapping** (loaded for real):
```
index.d == 1024        ✓
index.ntotal == 286629 ✓  (matches spec exactly)
len(global_ids) == 286629 ✓
video_metadata rows == 873 ✓ (matches "873 unique videos")
checkpoint load: 0 missing keys, 0 unexpected keys ✓
column auto-detection: vector_id→'vector_id', video_id→'video_id',
  frame_id→'frame_id', frame_path→'frame_path', namespace→'parent_namespace',
  timestamp→ not found (real column is 'timestamp_s', 100% null anyway)
```

**Query encoder**, three example queries — all produced shape `(1, 1024)`,
`float32`, all-finite, L2 norm `1.000000`, ~0.25–0.35s/query on CPU.

**Real API smoke test** (actual running `python main.py` server, real HTTP
requests, not `TestClient`):

```
Query: "a person riding a motorcycle on a street"
  → 200 OK, top score 0.6745, vector_id=248085, video_id=L28_V001

Query: "a television news presenter in a studio"
  → 200 OK, top score 0.7249, vector_id=19739, video_id=L21_V031

Query: "people sitting together around a table"
  → 200 OK, top 5 results all consecutive frames from video L28_V011
    (a real conversation-around-a-table scene — sensible retrieval,
    not noise)
```
All scores real floats, correctly sorted descending, every `vector_id`
correctly mapped to its metadata row. Search latency ~0.3–0.5s/query after
the one-time ~11–25s model+index load.

**UI test** — backend + frontend both launched for real (`python main.py`,
`npm run dev`), driven with a real headless-Chromium session via Playwright
(no `chromium-cli` available in this environment; Node/Playwright were
installed for the session). Result: UI correctly shows
"Backend: ONLINE / Engine: LIVE · FASTAPI" (not the mock fallback), submitting
"a person riding a motorcycle on a street" returned 24 real ranked cards with
descending scores (67% → 66%) and real video/frame IDs, matching the `curl`
test exactly. Two cosmetic, out-of-scope gaps observed and *not* fixed
(frontend code, not part of this task):
- No thumbnail images render — BEiT3 responses carry `frame_path` (a
  relative path), not base64 image bytes; we only have embeddings, not the
  keyframe image files themselves.
- The result card's time/folder label shows "UNKNOWN" — the frontend's
  field-mapping (`folder_key`/`split`) doesn't have an equivalent for the
  new `namespace` field.

Separately, the UI's right-side "Q&A" **chat copilot** panel (distinct from
the left-side "Q&A" *search* tab) always shows "DEMO copilot · mock answers,
no backend" — this is pre-existing, intentional, and unrelated to this
change (`ChatPanel.jsx`'s `sendChat()` calls a local `askCopilot()` mock
function and never makes an HTTP request; the earlier "Connect UI" PR
explicitly scoped this out — *"Non-Goals: Replace the mock chat copilot with
a backend service"*).

---

## Required `.env` variables (deployment machine)

```env
BEIT3_FAISS_INDEX_PATH=/absolute/path/beit3_faiss.index
BEIT3_GLOBAL_IDS_PATH=/absolute/path/global_ids.parquet
BEIT3_VIDEO_METADATA_PATH=/absolute/path/video_metadata.parquet
BEIT3_INDEX_META_PATH=/absolute/path/index_meta.json
BEIT3_CHECKPOINT_PATH=/absolute/path/beit3_large_patch16_384_f30k_retrieval.pth
BEIT3_TOKENIZER_PATH=/absolute/path/beit3.spm
BEIT3_DEVICE=cuda   # or cpu; falls back to cpu automatically if cuda unavailable
```

Optional, only if `global_ids.parquet`'s real columns don't match
auto-detection: `BEIT3_COL_VECTOR_ID`, `BEIT3_COL_VIDEO_ID`,
`BEIT3_COL_FRAME_ID`, `BEIT3_COL_FRAME_PATH`, `BEIT3_COL_TIMESTAMP`,
`BEIT3_COL_NAMESPACE`.

None of the checkpoint/index/parquet files are committed to Git — they must
be placed on each machine that runs the backend, per `.gitignore`.

---

## Automated Test Results

**Command executed:** `python -m pytest tests/test_beit3_retriever.py -v`

```text
============================= test session starts =============================
collected 13 items

tests/test_beit3_retriever.py::TokenizerOffsetMappingTests::test_tokenize_truncates_long_queries PASSED
tests/test_beit3_retriever.py::TokenizerOffsetMappingTests::test_tokenize_wraps_with_bos_eos_and_pads PASSED
tests/test_beit3_retriever.py::TokenizerOffsetMappingTests::test_unk_piece_maps_to_reserved_unk_id PASSED
tests/test_beit3_retriever.py::ColumnDetectionTests::test_detect_columns_fails_loudly_when_vector_id_missing PASSED
tests/test_beit3_retriever.py::ColumnDetectionTests::test_detect_columns_respects_explicit_override PASSED
tests/test_beit3_retriever.py::ColumnDetectionTests::test_detect_columns_uses_candidates PASSED
tests/test_beit3_retriever.py::ColumnDetectionTests::test_first_matching_column_case_insensitive PASSED
tests/test_beit3_retriever.py::SearchVisualIntegrationTests::test_search_visual_rejects_invalid_top_k PASSED
tests/test_beit3_retriever.py::SearchVisualIntegrationTests::test_search_visual_returns_real_scores_and_metadata PASSED
tests/test_beit3_retriever.py::QueryVectorValidationTests::test_accepts_normalized_vector PASSED
tests/test_beit3_retriever.py::QueryVectorValidationTests::test_rejects_non_finite PASSED
tests/test_beit3_retriever.py::QueryVectorValidationTests::test_rejects_unnormalized_vector PASSED
tests/test_beit3_retriever.py::QueryVectorValidationTests::test_rejects_wrong_shape PASSED

======================= 13 passed, 17 warnings in ~6-8s =======================
```

(Warnings are an unrelated `torch.jit.script` deprecation notice from a
third-party dependency, not from this code.)

---

## Files changed

| File | Change |
|---|---|
| `src/services/beit3_retriever.py` | new — retrieval service |
| `src/utils/beit3_backbone.py` | new — vendored BEiT3 architecture |
| `tests/test_beit3_retriever.py` | new — unit tests |
| `src/services/user_service.py` | `getImageDataSingleTextSearch` now uses BEiT3 |
| `src/services/fusion_service.py` | removed fake rank-derived score |
| `src/config/settings.py` | added `BEIT3_*` settings fields |
| `.env.example` | documented `BEIT3_*` env vars |
| `requirements.txt` | added `pyarrow`, `sentencepiece`, `torchscale` |
| `.gitignore` | ignore BEiT3/FAISS runtime artifacts |
