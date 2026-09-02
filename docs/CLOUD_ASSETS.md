# Cloud asset storage — team workflow

The backend can read its runtime artifacts (retrieval FAISS index, parquet
files, model checkpoint/tokenizer where applicable, media-info,
map-keyframes) and keyframes from **Azure Blob** or an **S3-compatible**
bucket instead of local disk. Only the app *reads*; publishing the dataset +
manifest is a one-time operator task.

Two independent retrieval backends can be served this way — **BEiT3** and
**Jina CLIP v2** (see `RETRIEVAL_BACKEND` in `.env.example`). They are two
different embedding spaces built from two different FAISS indexes; a sync
only ever pulls the artifacts for the *active* backend, never both (see
[Selective sync](#selective-sync-by-backend) below).

## The manifest

Everything is driven by a single versioned file, **`hcmai-assets.json`**, in
the **metadata** container/bucket:

```json
{
  "version": "2026-09-01",
  "artifacts": [
    {"name": "faiss_index", "container": "embeddings", "key": "beit3/beit3_faiss.index",
     "size": 1176325506, "sha256": "…64 hex…"},
    {"name": "global_ids",  "container": "embeddings", "key": "beit3/global_ids.parquet",
     "size": 4789145, "sha256": "…"},
    // checkpoint, tokenizer, video_metadata, index_meta, media_info, map_keyframes …

    {"name": "jina_faiss_index", "container": "embeddings",
     "key": "indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/jina_faiss.index",
     "size": 2844580986, "sha256": "…64 hex…"},
    {"name": "jina_global_ids", "container": "embeddings",
     "key": "indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/global_ids.parquet",
     "size": 10721963, "sha256": "…"},
    {"name": "jina_video_metadata", "container": "embeddings",
     "key": "indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/video_metadata.parquet",
     "size": 22265, "sha256": "…"},
    {"name": "jina_index_meta", "container": "embeddings",
     "key": "indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/index_meta.json",
     "size": 249, "sha256": "…"}
  ],
  "keyframes": {"container": "keyframes", "prefix": "", "layout": "{namespace}/{video_id}/{frame_id}.webp"}
}
```

> The Jina index above is **already built and uploaded** on the shared account
> (693,124 vectors / 873 videos, 1024-d, `fine_keyframes_jina_clip_v2_1024d_v2`).
> Its per-blob sizes + SHA-256 are stamped in each blob's metadata, so a
> ready-to-publish `hcmai-assets.json` can be generated without downloading the
> ~2.65 GiB index — regenerate it from blob properties and upload it once to
> `metadata/hcmai-assets.json`.

`sync_artifacts` streams each blob, checks **size + SHA-256**, and only promotes
a version once **every requested artifact** is present and verified — a
half-finished sync never shadows a good one, and (see below) a sync scoped to
one backend's artifacts is not blocked by the other backend's artifacts also
listed in the manifest. Keyframes are fetched on demand from the `keyframes`
container and LRU-cached locally.

## Jina CLIP v2 artifact set

| Artifact name (manifest) | Published blob (key basename) | Contains |
| --- | --- | --- |
| `jina_faiss_index` | `jina_faiss.index` | `IndexIDMap2(IndexFlatIP(1024))` |
| `jina_global_ids` | `global_ids.parquet` | one row per keyframe (schema below) |
| `jina_video_metadata` | `video_metadata.parquet` | per-video: `video_id`, `parent_namespace`, `frame_count`, `embedding_dim`, `first_vector_id`, `artifact_blob` |
| `jina_index_meta` | `index_meta.json` | `embedding_run`, `model`, `embedding_dim`, `metric`, `vector_count`, `video_count` |

**`jina_global_ids` parquet — two accepted schemas.** `JinaRetriever` reads
either and normalizes internally (`_normalize_global_ids`):

* **Canonical** (produced by `scripts/cloud/build_jina_index.py`): `vector_id`,
  `split`, `video_id`, `embedding_row`, `keyframe_ordinal`, `timestamp_ms`,
  `asset_key`, `frame_path` (alias of `asset_key`), `source_frame_id`.
* **Azure merge schema** (what is published today): `parent_namespace`,
  `video_id`, `frame_id`, `frame_path`, `timestamp` (seconds), `source_fps`,
  `source_frame_idx`, `local_position`, `vector_id`. Here `frame_path` is
  already the literal keyframe key, so it is used as `asset_key`.

`asset_key` (or, in the merge schema, `frame_path`) is the real, literal
keyframe object key inside the `keyframes` container, e.g.
`L21_a/L21_V001/keyframe_0000.jpg` — never a path guessed from a numeric frame
id. The keyframe resolver (`src/services/assets/__init__.py`) looks at
`asset_key` first, then `frame_path`, before falling back to anything else
(see [Keyframe resolution](#keyframe-resolution)).

There is no `jina_checkpoint`/`jina_tokenizer` artifact: the Jina CLIP v2
model itself is loaded from a **local, pinned** HuggingFace snapshot
(`JINA_MODEL_PATH` + `JINA_MODEL_REVISION`, `JINA_LOCAL_FILES_ONLY=true`), not
synced through this manifest.

### Azure object layout (as produced by the existing embedding pipeline)

The Kaggle notebooks under `scripts/notebooks/` (`embed-jina-upload-azure-*`,
`merge-azure-jina-embedding-index.ipynb`) already produce this layout in the
`embeddings` container:

```text
embeddings/<embedding_run>/jina/<namespace>/<video_id>.npy       # per-video (N,1024) float16
embeddings/<embedding_run>/records/<namespace>/<video_id>.json   # per-video {"records": [...]}
embeddings/checkpoints/<embedding_run>/<part>/<ns>/<vid>.json    # per-video resume marker + sha256
embeddings/indexes/<embedding_run>/jina/jina_faiss.index         # merged/built index
embeddings/indexes/<embedding_run>/jina/global_ids.parquet
embeddings/indexes/<embedding_run>/jina/video_metadata.parquet
embeddings/indexes/<embedding_run>/jina/index_meta.json
metadata/map-keyframes/<video_id>.csv                            # n,pts_time,fps,frame_idx (n is 1-based)
```

Keyframes themselves live in the `keyframes` container at
`<namespace>/<video_id>/<frame_file>` (e.g. `L21_a/L21_V001/keyframe_0000.jpg`,
namespaces `L21_a`…`L30_a`) — this is exactly the value stored in `frame_path`
/ `asset_key`.

## Go live on the existing Jina index (no rebuild)

The `fine_keyframes_jina_clip_v2_1024d_v2` index is already published. To use it:

1. Generate `hcmai-assets.json` for it (sizes + SHA-256 come from blob
   metadata, so the ~2.65 GiB index is never downloaded to build the
   manifest), then upload it once to `metadata/hcmai-assets.json`.
2. Per member: `CLOUD_ASSETS_ENABLED=true`, `CLOUD_ASSETS_PROVIDER=azure_blob`,
   `AZURE_STORAGE_CONNECTION_STRING=…`, `RETRIEVAL_BACKEND=jina_clip_v2`,
   `JINA_MODEL_PATH=jinaai/jina-clip-v2`, `JINA_MODEL_REVISION=<pinned sha>`.
   Save → restart → Cloud Assets tab → **Sync artifacts** (pulls only the four
   `jina_*` artifacts) → the retriever picks up the synced FAISS index +
   parquet automatically.

## (Re)build the Jina runtime artifacts

Only needed for a *new* corpus (different keyframes / model / preprocessing).
`scripts/cloud/build_jina_index.py` reads the per-video `.npy` + records JSON
(downloaded locally from the `embeddings` container ahead of time) and an
optional map-keyframes directory, validates them, and writes the artifacts
plus a build report in the **canonical** schema. It never loads the whole
corpus into RAM — each video's `.npy` is opened with `mmap_mode='r'` and added
to the FAISS index one video at a time.

```bash
python scripts/cloud/build_jina_index.py \
    --embeddings-root /data/jina/embeddings \
    --records-root /data/jina/records \
    --map-keyframes-root /data/map-keyframes \
    --model-id jinaai/jina-clip-v2 \
    --model-revision <pinned-commit-sha> \
    --embedding-run fine_keyframes_jina_clip_v2_1024d_v2 \
    --out-dir ./jina_runtime
```

Validation performed (fails loudly, never silently drops a bad row):
embedding shape `(N, 1024)`; finite values; L2-normalized (±2e-3); `.npy` row
order matches the metadata's `local_position` exactly; unique `vector_id`
and `asset_key` across the whole corpus; non-empty, path-safe `video_id` /
`split`; a resolvable `timestamp_ms` for every row (record value, or a
map-keyframes fallback — never invented); FAISS `ntotal` equal to the
parquet row count.

Then publish the manifest for it the same way as BEiT3's:

```bash
# add the jina_* entries (see manifest_spec.example.json) pointing "local" at
# ./jina_runtime/jina_faiss.index, jina_global_ids.parquet, jina_index_meta.json
python scripts/cloud/build_asset_manifest.py --spec manifest_spec.json --out hcmai-assets.json
```

## Selective sync by backend

`POST /settings/cloud/sync` syncs only the **active** backend's artifacts
(`BACKEND_ARTIFACT_NAMES` in `src/services/assets/base.py`) when the request
doesn't name specific artifacts — a member running `RETRIEVAL_BACKEND=jina_clip_v2`
is never made to also download the (larger) BEiT3 checkpoint + FAISS index,
and vice versa. Promotion to *current* is gated on the requested subset being
fully present with a verified size + SHA-256 (`ArtifactCache.is_version_verified`),
not merely "the file exists" and not on every artifact the manifest happens
to list.

## Keyframe resolution

`resolve_keyframe_file` (and the underlying `_keyframe_rel_path`) resolve a
search-result item to a cache key in this order:

1. `asset_key` — the authoritative cloud key, if the item carries one.
2. `frame_path` / `image_path` / `keyframe_path`.
3. The manifest's `keyframes.layout` template, only if it uses solely the
   whitelisted placeholders (`namespace`, `split`, `video_id`, `frame_id`,
   `frame_name`) and every value it needs is present — a malformed or
   attacker-influenced layout is ignored, never used to build a path.
4. The legacy `video_id`/`split`/`frame_id` heuristic BEiT3 result rows have
   always used (produces a `.webp` name).

`.jpg`, `.jpeg`, `.webp` and `.png` are all recognised. Nothing here ever
reformats a numeric frame id into a guessed filename like
`keyframe_0000.jpg` — that filename only ever comes from real `asset_key`/
`frame_path` mapping data or a validated `layout`. Both the artifact cache and
the keyframe LRU cache reject any path that would resolve outside their root
(`../` traversal, absolute-path escapes) rather than reading/writing it.

## Migration / rollback: BEiT3 ↔ Jina CLIP v2

Switching backends is a config-only change — no code deploy:

1. Sync the target backend's artifacts (Settings → Cloud Assets → **Sync
   artifacts**, or `POST /settings/cloud/sync`) if not already local.
2. Set `RETRIEVAL_BACKEND` to `beit3` or `jina_clip_v2` in Settings →
   Retrieval (or `.env`) and restart.
3. Textual KIS, grounded Q&A candidate retrieval, and TRAKE per-event
   retrieval now come from the new backend. Image-similarity endpoints
   ('Similar' on a capture, search-by-uploaded-image) are unaffected — they
   always use BEiT3, so its artifacts should stay configured/synced even
   while `RETRIEVAL_BACKEND=jina_clip_v2`.
4. Roll back by setting `RETRIEVAL_BACKEND` back to `beit3` and restarting;
   nothing about the BEiT3 artifacts or index is touched by running with
   Jina active, so this is always safe.

Because the two backends' FAISS indexes, vector-id spaces, and result rows
never mix, a rollback (or a forward migration) is always a clean cut — there
is no partial/mixed state to reconcile.

## 1. Build & upload the artifacts (operator, once per dataset version)

Upload your built runtime files to the storage account. Any layout works — you
name the keys in the manifest. Example (Azure):

```bash
az storage blob upload-batch -d embeddings -s ./beit3_runtime \
  --pattern "*.index" --pattern "*.parquet" --pattern "index_meta.json" \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
az storage blob upload -c metadata -n beit3/beit3.spm -f ./beit3_runtime/beit3.spm --overwrite ...
```

Keyframes go under the `keyframes` container as `<namespace>/<video_id>/<frame>` —
matching the `frame_path` column in `global_ids.parquet` (adjust the manifest
`keyframes.prefix` if your keys have an extra leading segment).

## 2. Generate `hcmai-assets.json`

```bash
cp scripts/cloud/manifest_spec.example.json manifest_spec.json
# edit every "local" path to your built files and the "key" to the blob key you used

python scripts/cloud/build_asset_manifest.py --spec manifest_spec.json --out hcmai-assets.json
```

It hashes each **local** file (fast — you already have them), validates the
result, and writes `hcmai-assets.json`. Add `--upload` to push it straight to
`<metadata>/hcmai-assets.json` (Azure connection string from
`--connection-string`, `AZURE_STORAGE_CONNECTION_STRING`, or the runtime config
store); otherwise it prints the `az` / `aws` command to run.

Re-run with a new `--version` whenever an artifact changes.

## 3. Each teammate enables cloud mode

In **Settings → Configuration** (or `.env` on first run):

| Field | Value |
| --- | --- |
| `CLOUD_ASSETS_ENABLED` | `true` |
| `CLOUD_ASSETS_PROVIDER` | `azure_blob` or `s3_compatible` |
| `AZURE_STORAGE_CONNECTION_STRING` *(Azure)* | the connection string (no wrapping quotes) |
| `S3_ENDPOINT_URL` / `S3_BUCKET` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` *(S3)* | … |
| `CLOUD_ASSETS_MANIFEST_KEY` | `hcmai-assets.json` (default) |

Save → Restart. Then on the **Cloud Assets** tab: **Test connection** →
**Load manifest** → **Sync artifacts**. The synced FAISS index / checkpoint /
parquet are used automatically in place of the local `BEIT3_*` paths;
keyframes stream on demand.

## Notes

* `azure-storage-blob` (Azure) / `boto3` (S3) must be installed — the Cloud
  Assets tab shows an SDK-availability row.
* The synced-artifact cache lives at `CLOUD_ASSETS_CACHE_PATH`
  (default `<app-data>/HCMAI2026/assets-cache`); the keyframe LRU cap is
  `CLOUD_ASSETS_KEYFRAME_CACHE_MAX_BYTES`.
* Loading a multi-GB FAISS index + model still needs enough local RAM — cloud
  mode changes where the files come from, not how much memory the search uses.
