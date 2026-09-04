# Cloud asset storage — team workflow

The backend can read its runtime artifacts (BEiT3 FAISS index, parquet files,
model checkpoint, tokenizer, media-info, map-keyframes) and keyframes from
**Azure Blob** or an **S3-compatible** bucket instead of local disk. Only the
app *reads*; publishing the dataset + manifest is a one-time operator task.

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
     "size": 4789145, "sha256": "…"}
    // checkpoint, tokenizer, video_metadata, index_meta, media_info, map_keyframes …
  ],
  "keyframes": {"container": "keyframes", "prefix": "", "layout": "{namespace}/{video_id}/{frame_id}.webp"}
}
```

`sync_artifacts` streams each blob, checks **size + SHA-256**, and only promotes
a version once **every** listed artifact is present — a half-finished sync never
shadows a good one. Keyframes are fetched on demand from the `keyframes`
container and LRU-cached locally.

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

## Jina fine-keyframe runtime

The final Jina artifacts live under:

```text
embeddings/indexes/fine_keyframes_jina_clip_v2_1024d_v2/jina/
```

Publish a separate manifest from the size/SHA-256 blob metadata written by the
merge notebook. This does not download the FAISS index:

```powershell
python scripts\cloud\publish_jina_manifest_from_azure.py --upload
```

Then configure and restart:

```env
CLOUD_ASSETS_ENABLED=true
CLOUD_ASSETS_PROVIDER=azure_blob
CLOUD_ASSETS_MANIFEST_KEY=hcmai-assets-jina.json
VISUAL_RETRIEVER=jina
JINA_MODEL_NAME_OR_PATH=jinaai/jina-clip-v2
JINA_MODEL_REVISION=
JINA_DEVICE=cpu
JINA_TRUNCATE_DIM=1024
JINA_QUERY_TASK=retrieval.query
```

In Settings -> Cloud Assets, run **Test connection**, **Load manifest**, and
**Sync artifacts** before switching `VISUAL_RETRIEVER` to `jina`. The Jina
retriever resolves checksum-verified artifact names prefixed with `jina_`.
BEiT3 remains an explicit rollback by setting `VISUAL_RETRIEVER=beit3`;
startup errors never silently cross embedding spaces.
* Loading a multi-GB FAISS index + model still needs enough local RAM — cloud
  mode changes where the files come from, not how much memory the search uses.
