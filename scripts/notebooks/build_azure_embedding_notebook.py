"""Generate the Kaggle worker notebook for disk-safe Azure embedding jobs.

This is intentionally a generator: keeping the code cells as Python strings makes
the notebook reviewable in git while avoiding a hand-maintained JSON document.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "embed-jina-upload-azure-5jobs-disk-safe.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(True),
    }


cells = [
    markdown(
        """
        # AIC 2026 - Jina CLIP v2 embedding with disk-safe Azure upload

        This is a **worker notebook** for one of five independent Kaggle jobs.
        It reads existing keyframes, builds embeddings per video, uploads each completed
        video immediately to Azure, verifies the remote blob, then removes the local NPY.
        Images and Vietnamese/English text share the same Jina CLIP v2 vector space.

        Output layout in the dedicated `embeddings` Azure container:

        ```text
        <embedding_run>/jina/<namespace>/<video_id>.npy
        <embedding_run>/records/<namespace>/<video_id>.json
        checkpoints/<embedding_run>/<part>/<namespace>/<video_id>.json
        ```

        This notebook builds normalized 1024-dimensional image vectors. The merger creates
        a separate Jina FAISS index; the current BEiT3 backend cannot consume it until a
        matching Jina text-query retriever is enabled.
        """
    ),
    code(
        """
        # Kaggle dependencies. Pin only the packages that are not part of the base image.
        !pip install -q azure-storage-blob pyarrow "transformers>=4.45,<5" sentencepiece einops timm
        """
    ),
    code(
        """
        # 1. Configuration. Make five copies of this notebook and change only JOB_ID.
        from pathlib import Path
        import hashlib
        import json
        import os

        JOB_ID = 1  # 1..5
        JOB_PRESETS = {
            1: ['L21_a', 'L22_a', 'L23_a'],
            2: ['L24_a', 'L25_a', 'L26_a'],
            3: ['L26_b', 'L26_c', 'L26_d'],
            4: ['L26_e', 'L27_a', 'L28_a'],
            5: ['L29_a', 'L30_a'],
        }
        EXCLUDED_FOLDERS = {'L25_b', 'L25_a1'}
        if JOB_ID not in JOB_PRESETS:
            raise ValueError('JOB_ID must be in 1..5')

        PART_NAME = f'part_{JOB_ID:02d}'
        ASSIGNED_FOLDERS = [x for x in JOB_PRESETS[JOB_ID] if x not in EXCLUDED_FOLDERS]

        # Set to 'azure' to stream keyframes directly from Azure one video at a time.
        # Use 'kaggle_input' only when the full keyframe corpus was added as a Kaggle Input.
        KEYFRAME_SOURCE = 'azure'
        AZURE_CONTAINER_KEYFRAMES = 'keyframes'
        AZURE_KEYFRAMES_PREFIX = ''  # Example: 'Keyframes' if blob paths have that leading folder.
        KEYFRAMES_ROOT = Path('/kaggle/input/aic-fine-keyframes/Keyframes')
        # Optional in Kaggle-input mode; fetched from Azure in azure mode. Files: <video_id>.csv.
        MAP_KEYFRAMES_ROOT = Path('/kaggle/input/aic-fine-keyframes/map-keyframes')

        # With Kaggle Internet enabled, the model is downloaded once from Hugging Face.
        # For offline runs, add a complete model snapshot as a Kaggle Input and set this
        # to its directory, for example '/kaggle/input/jina-clip-v2'.
        JINA_MODEL_ID = 'jinaai/jina-clip-v2'
        JINA_MODEL_PATH = ''
        JINA_LOCAL_FILES_ONLY = bool(JINA_MODEL_PATH)
        JINA_TRUNCATE_DIM = 1024
        ENABLE_JINA = True
        DEVICE = 'cuda'
        JINA_BATCH_SIZE = 8  # Conservative T4 default for the 0.9B model at 512px.
        IMAGE_DECODE_WORKERS = 8
        MAX_VIDEOS_TO_PROCESS = None  # Set to 1 for a smoke test.

        # This identifies one compatible corpus. Do not reuse a run name after changing
        # source keyframes, model checkpoint, image preprocessing, or timestamp mapping.
        EMBEDDING_RUN = 'fine_keyframes_jina_clip_v2_1024d_v2'
        AZURE_CONTAINER_METADATA = 'metadata'
        AZURE_CONTAINER_EMBEDDINGS = 'embeddings'
        AZURE_UPLOAD_ENABLED = True
        DELETE_LOCAL_ARTIFACTS_AFTER_UPLOAD = True
        AZURE_DOWNLOAD_CONCURRENCY = 16
        AZURE_DOWNLOAD_RETRIES = 4
        PREFETCH_NEXT_VIDEO = True
        EMBEDDING_STORAGE_DTYPE = 'float16'  # FAISS merger converts back to float32.

        WORK_ROOT = Path('/kaggle/working') / f'{EMBEDDING_RUN}_{PART_NAME}'
        KEYFRAME_CACHE_ROOT = WORK_ROOT / 'keyframe-cache'
        LOCAL_EMBEDDINGS_ROOT = WORK_ROOT / 'embeddings'
        LOCAL_RECORDS_ROOT = WORK_ROOT / 'records'
        LOCAL_CHECKPOINT_ROOT = WORK_ROOT / 'checkpoints'
        for path in (LOCAL_EMBEDDINGS_ROOT, LOCAL_RECORDS_ROOT, LOCAL_CHECKPOINT_ROOT, KEYFRAME_CACHE_ROOT):
            path.mkdir(parents=True, exist_ok=True)

        run_signature = {
            'embedding_run': EMBEDDING_RUN,
            'jina_model': JINA_MODEL_ID,
            'jina_truncate_dim': JINA_TRUNCATE_DIM,
            'storage_dtype': EMBEDDING_STORAGE_DTYPE,
            'keyframes_root': str(KEYFRAMES_ROOT),
        }
        CONFIG_FINGERPRINT = hashlib.sha256(
            json.dumps(run_signature, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]
        if KEYFRAME_SOURCE not in {'azure', 'kaggle_input'}:
            raise ValueError("KEYFRAME_SOURCE must be 'azure' or 'kaggle_input'")
        print('Part:', PART_NAME, 'folders:', ASSIGNED_FOLDERS, 'source:', KEYFRAME_SOURCE)
        print('Embedding run:', EMBEDDING_RUN, 'fingerprint:', CONFIG_FINGERPRINT)
        """
    ),
    code(
        """
        # 2. Azure connection. Add AZURE_STORAGE_CONNECTION_STRING as a Kaggle Secret.
        from azure.storage.blob import BlobServiceClient, ContentSettings

        def get_secret(name):
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return os.environ.get(name)

        connection_string = get_secret('AZURE_STORAGE_CONNECTION_STRING')
        if AZURE_UPLOAD_ENABLED and not connection_string:
            raise RuntimeError('Missing Kaggle Secret AZURE_STORAGE_CONNECTION_STRING')

        blob_service = BlobServiceClient.from_connection_string(connection_string) if AZURE_UPLOAD_ENABLED else None
        metadata_container = blob_service.get_container_client(AZURE_CONTAINER_METADATA) if blob_service else None
        embeddings_container = blob_service.get_container_client(AZURE_CONTAINER_EMBEDDINGS) if blob_service else None
        container = embeddings_container
        keyframes_container = blob_service.get_container_client(AZURE_CONTAINER_KEYFRAMES) if blob_service else None
        if metadata_container:
            metadata_container.get_container_properties()
            print('Azure container is reachable:', AZURE_CONTAINER_METADATA)
            embeddings_container.get_container_properties()
            print('Azure container is reachable:', AZURE_CONTAINER_EMBEDDINGS)
        if KEYFRAME_SOURCE == 'azure':
            keyframes_container.get_container_properties()
            print('Azure keyframes container is reachable:', AZURE_CONTAINER_KEYFRAMES)
        """
    ),
    code(
        """
        # 3. Discover only the videos assigned to this part.
        import csv
        import re
        from collections import defaultdict

        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
        FRAME_NUMBER = re.compile(r'(?:keyframe_|frame_)?(\\d+)$', re.IGNORECASE)

        def frame_sort_key(path):
            match = FRAME_NUMBER.search(path.stem)
            return (int(match.group(1)) if match else 10**12, path.name)

        def discover_local_jobs():
            jobs = []
            for namespace in ASSIGNED_FOLDERS:
                root = KEYFRAMES_ROOT / namespace
                if not root.exists():
                    print('WARNING: keyframe namespace not found:', root)
                    continue
                for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                    frames = sorted(
                        [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                        key=frame_sort_key,
                    )
                    if frames:
                        jobs.append({'namespace': namespace, 'video_id': video_dir.name, 'frames': frames})
            return jobs

        def discover_azure_jobs():
            grouped = defaultdict(list)
            for namespace in ASSIGNED_FOLDERS:
                prefix = '/'.join(x for x in (AZURE_KEYFRAMES_PREFIX.strip('/'), namespace) if x) + '/'
                for blob in keyframes_container.list_blobs(name_starts_with=prefix):
                    relative = blob.name[len(prefix):]
                    parts = relative.split('/')
                    if len(parts) != 2 or Path(parts[1]).suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    grouped[(namespace, parts[0])].append({
                        'name': blob.name,
                        'size': int(blob.size or 0),
                    })
            jobs = []
            for (namespace, video_id), blob_items in sorted(grouped.items()):
                jobs.append({
                    'namespace': namespace,
                    'video_id': video_id,
                    'frame_blobs': sorted(blob_items, key=lambda item: frame_sort_key(Path(item['name']))),
                })
            return jobs

        jobs = discover_azure_jobs() if KEYFRAME_SOURCE == 'azure' else discover_local_jobs()
        if MAX_VIDEOS_TO_PROCESS is not None:
            jobs = jobs[:int(MAX_VIDEOS_TO_PROCESS)]
        print('Videos assigned:', len(jobs), 'frames:', sum(len(x.get('frames', x.get('frame_blobs', []))) for x in jobs))
        for job in jobs[:5]:
            print(job['namespace'], job['video_id'], len(job.get('frames', job.get('frame_blobs', []))))
        if not jobs:
            raise RuntimeError('No keyframes found. Check KEYFRAMES_ROOT and namespace layout.')
        """
    ),
    code(
        """
        # 4. Timestamp/record helpers. n in map CSV is one-based; keyframe_0000 maps to n=1.
        import numpy as np
        from PIL import Image

        def atomic_write_bytes(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)

        def atomic_write_json(path, payload):
            atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + '\\n').encode('utf-8'))

        def load_timestamp_map(video_id, root=None):
            path = (root or MAP_KEYFRAMES_ROOT) / f'{video_id}.csv'
            values = {}
            if not path.exists():
                return values
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    try:
                        values[int(row['n'])] = {
                            'timestamp': float(row['pts_time']),
                            'source_fps': float(row.get('fps') or 0) or None,
                            'source_frame_idx': int(float(row.get('frame_idx') or 0)),
                        }
                    except (KeyError, TypeError, ValueError):
                        continue
            return values

        def fetch_azure_timestamp_map(video_id):
            # Map files are small and preserve exact pts_time/source frame alignment.
            local_path = KEYFRAME_CACHE_ROOT / 'map-keyframes' / f'{video_id}.csv'
            if not local_path.exists():
                blob = metadata_container.get_blob_client(f'map-keyframes/{video_id}.csv')
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_bytes(local_path, blob.download_blob().readall())
                except Exception:
                    return {}
            return load_timestamp_map(video_id, root=local_path.parent)

        def prepare_job_frames(job):
            if KEYFRAME_SOURCE == 'kaggle_input':
                return job['frames']
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time

            namespace, video_id = job['namespace'], job['video_id']
            local_dir = KEYFRAME_CACHE_ROOT / namespace / video_id
            local_dir.mkdir(parents=True, exist_ok=True)

            def download_one(blob_item):
                blob_name = blob_item['name']
                expected_size = int(blob_item.get('size') or 0)
                destination = local_dir / Path(blob_name).name
                if destination.exists() and destination.stat().st_size > 0:
                    if not expected_size or destination.stat().st_size == expected_size:
                        return destination
                temporary = destination.with_suffix(destination.suffix + '.part')
                last_error = None
                for attempt in range(AZURE_DOWNLOAD_RETRIES):
                    try:
                        with temporary.open('wb') as handle:
                            keyframes_container.get_blob_client(blob_name).download_blob(
                                max_concurrency=1,
                            ).readinto(handle)
                        actual_size = temporary.stat().st_size
                        if actual_size <= 0 or (expected_size and actual_size != expected_size):
                            raise RuntimeError(
                                f'Blob size mismatch for {blob_name}: {actual_size} != {expected_size}'
                            )
                        temporary.replace(destination)
                        return destination
                    except Exception as exc:
                        last_error = exc
                        if temporary.exists():
                            temporary.unlink()
                        if attempt + 1 < AZURE_DOWNLOAD_RETRIES:
                            time.sleep(min(8, 2 ** attempt))
                raise RuntimeError(f'Cannot download {blob_name}') from last_error

            frames = []
            with ThreadPoolExecutor(max_workers=AZURE_DOWNLOAD_CONCURRENCY) as pool:
                futures = [pool.submit(download_one, item) for item in job['frame_blobs']]
                for completed_count, future in enumerate(as_completed(futures), 1):
                    frames.append(future.result())
                    if completed_count % 500 == 0:
                        print(f'Downloaded {completed_count}/{len(futures)} frames for {video_id}')
            return sorted(frames, key=frame_sort_key)

        def cleanup_job_frames(namespace, video_id):
            if KEYFRAME_SOURCE == 'azure':
                import shutil
                shutil.rmtree(KEYFRAME_CACHE_ROOT / namespace / video_id, ignore_errors=True)

        def record_for_frame(namespace, video_id, frame_path, position, timestamp_map):
            match = FRAME_NUMBER.search(frame_path.stem)
            file_index = int(match.group(1)) if match else position
            # Extraction names begin at 0000, map n begins at 1.
            mapping = timestamp_map.get(file_index + 1, {})
            return {
                'parent_namespace': namespace,
                'video_id': video_id,
                'frame_id': frame_path.stem,
                'frame_path': f'{namespace}/{video_id}/{frame_path.name}',
                'timestamp': mapping.get('timestamp'),
                'source_fps': mapping.get('source_fps'),
                'source_frame_idx': mapping.get('source_frame_idx'),
                'local_position': position,
            }

        def open_rgb(path):
            with Image.open(path) as image:
                return image.convert('RGB')
        """
    ),
    code(
        """
        # 5. Load Jina CLIP v2. Image and query text must use this exact model/dimension.
        import gc
        import torch
        from tqdm.auto import tqdm

        if DEVICE == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but no GPU is available.')
        print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

        def load_jina():
            from transformers import AutoModel
            source = JINA_MODEL_PATH if JINA_MODEL_PATH else JINA_MODEL_ID
            model = AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                local_files_only=JINA_LOCAL_FILES_ONLY,
                torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            ).eval().to(DEVICE)
            def encode(images):
                with torch.inference_mode():
                    vectors = model.encode_image(images, truncate_dim=JINA_TRUNCATE_DIM)
                if isinstance(vectors, torch.Tensor):
                    vectors = vectors.detach().float().cpu().numpy()
                vectors = np.asarray(vectors, dtype='float32')
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms <= 0):
                    raise RuntimeError('Jina returned a zero-norm image embedding')
                return vectors / norms
            return model, encode

        def unload(model):
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def encode_paths(paths, encode_images, initial_batch_size):
            from concurrent.futures import ThreadPoolExecutor
            chunks = []
            pos, batch_size = 0, initial_batch_size
            with ThreadPoolExecutor(max_workers=IMAGE_DECODE_WORKERS) as decode_pool:
                while pos < len(paths):
                    batch_paths = paths[pos:pos + batch_size]
                    try:
                        images = list(decode_pool.map(open_rgb, batch_paths))
                        vectors = encode_images(images)
                        if not np.isfinite(vectors).all():
                            raise RuntimeError('Encoder returned NaN/Inf')
                        chunks.append(vectors)
                        pos += len(batch_paths)
                    except torch.cuda.OutOfMemoryError:
                        if batch_size <= 1:
                            raise
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        print('CUDA OOM: reducing batch size to', batch_size)
            return np.concatenate(chunks, axis=0)
        """
    ),
    code(
        """
        # 6. Azure upload + per-video resume.
        import io

        def blob_name_for(model_name, namespace, video_id, suffix):
            return f'{EMBEDDING_RUN}/{model_name}/{namespace}/{video_id}{suffix}'

        def checkpoint_blob_name(namespace, video_id):
            return f'checkpoints/{EMBEDDING_RUN}/{PART_NAME}/{namespace}/{video_id}.json'

        def sha256_path(path):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()

        def remote_checkpoint(namespace, video_id):
            if not container:
                return None
            client = container.get_blob_client(checkpoint_blob_name(namespace, video_id))
            try:
                return json.loads(client.download_blob().readall())
            except Exception:
                return None

        def upload_verified(local_path, remote_name, content_type):
            if not container:
                return
            digest = sha256_path(local_path)
            size = local_path.stat().st_size
            client = container.get_blob_client(remote_name)
            try:
                props = client.get_blob_properties()
                if props.size == size and props.metadata.get('sha256') == digest:
                    return
                raise RuntimeError(f'Existing Azure blob differs: {remote_name}')
            except Exception as exc:
                if 'differs' in str(exc):
                    raise
            with local_path.open('rb') as handle:
                client.upload_blob(
                    handle,
                    overwrite=False,
                    metadata={'sha256': digest},
                    content_settings=ContentSettings(content_type=content_type),
                    max_concurrency=4,
                )
            props = client.get_blob_properties()
            if props.size != size or props.metadata.get('sha256') != digest:
                raise RuntimeError(f'Azure verification failed: {remote_name}')

        def save_npy_atomic(path, vectors):
            buffer = io.BytesIO()
            np.save(buffer, vectors, allow_pickle=False)
            atomic_write_bytes(path, buffer.getvalue())
        """
    ),
    code(
        """
        # 7. Run embeddings. Load Jina once, then overlap Azure download of video N+1
        # with GPU encoding of video N. At most two videos exist in the local cache.
        from concurrent.futures import ThreadPoolExecutor
        import time

        MODEL_SPECS = []
        if ENABLE_JINA:
            MODEL_SPECS.append(('jina', load_jina, JINA_BATCH_SIZE))
        if not MODEL_SPECS:
            raise RuntimeError('Enable at least one embedding model.')

        expected_models = [x[0] for x in MODEL_SPECS]
        completed, failed, pending_jobs = 0, [], []
        for job in jobs:
            namespace, video_id = job['namespace'], job['video_id']
            checkpoint = remote_checkpoint(namespace, video_id)
            if checkpoint and checkpoint.get('config_fingerprint') == CONFIG_FINGERPRINT and checkpoint.get('models') == expected_models:
                print('SKIP complete:', namespace, video_id)
                completed += 1
            else:
                pending_jobs.append(job)

        print('Already complete:', completed, '| Pending:', len(pending_jobs))
        loaded_models = {}
        prefetch_pool = ThreadPoolExecutor(max_workers=1) if KEYFRAME_SOURCE == 'azure' and PREFETCH_NEXT_VIDEO else None

        def schedule_download(job):
            if prefetch_pool is None:
                return None
            return prefetch_pool.submit(prepare_job_frames, job)

        current_download = schedule_download(pending_jobs[0]) if pending_jobs else None
        total_processed_frames = 0
        run_started = time.perf_counter()

        try:
            for model_name, loader, batch_size in MODEL_SPECS:
                print('Loading model once for the whole job:', model_name)
                model, encode_images = loader()
                loaded_models[model_name] = (model, encode_images, batch_size)
                print('Model ready:', model_name)

            for pending_index, job in enumerate(pending_jobs):
                namespace, video_id = job['namespace'], job['video_id']
                frames, next_download = [], None
                video_started = time.perf_counter()
                try:
                    download_started = time.perf_counter()
                    frames = current_download.result() if current_download is not None else prepare_job_frames(job)
                    download_seconds = time.perf_counter() - download_started

                    # Start downloading the next video before this video enters the GPU.
                    if pending_index + 1 < len(pending_jobs):
                        next_download = schedule_download(pending_jobs[pending_index + 1])

                    print(
                        f'\\n[{completed + 1}/{len(jobs)}] {namespace}/{video_id}: '
                        f'{len(frames)} frames, download wait {download_seconds:.1f}s'
                    )
                    timestamp_map = fetch_azure_timestamp_map(video_id) if KEYFRAME_SOURCE == 'azure' else load_timestamp_map(video_id)
                    records = [record_for_frame(namespace, video_id, path, i, timestamp_map) for i, path in enumerate(frames)]
                    artifacts = {}
                    encode_started = time.perf_counter()
                    for model_name, (model, encode_images, batch_size) in loaded_models.items():
                        vectors = encode_paths(frames, encode_images, batch_size)
                        if len(vectors) != len(records):
                            raise RuntimeError(f'{model_name}: vectors/records length mismatch')
                        stored_vectors = vectors.astype(EMBEDDING_STORAGE_DTYPE, copy=False)
                        output = LOCAL_EMBEDDINGS_ROOT / model_name / namespace / f'{video_id}.npy'
                        save_npy_atomic(output, stored_vectors)
                        upload_verified(output, blob_name_for(model_name, namespace, video_id, '.npy'), 'application/octet-stream')
                        artifacts[model_name] = {
                            'blob': blob_name_for(model_name, namespace, video_id, '.npy'),
                            'dimension': int(vectors.shape[1]),
                            'frame_count': int(vectors.shape[0]),
                            'storage_dtype': str(stored_vectors.dtype),
                            'sha256': sha256_path(output),
                        }
                        if DELETE_LOCAL_ARTIFACTS_AFTER_UPLOAD:
                            output.unlink()
                    encode_seconds = time.perf_counter() - encode_started

                    record_path = LOCAL_RECORDS_ROOT / namespace / f'{video_id}.json'
                    atomic_write_json(record_path, {'records': records})
                    record_blob = f'{EMBEDDING_RUN}/records/{namespace}/{video_id}.json'
                    upload_verified(record_path, record_blob, 'application/json')
                    if DELETE_LOCAL_ARTIFACTS_AFTER_UPLOAD:
                        record_path.unlink()

                    checkpoint = {
                        'status': 'completed',
                        'config_fingerprint': CONFIG_FINGERPRINT,
                        'embedding_run': EMBEDDING_RUN,
                        'part_name': PART_NAME,
                        'namespace': namespace,
                        'video_id': video_id,
                        'models': expected_models,
                        'records_blob': record_blob,
                        'frame_count': len(records),
                        'artifacts': artifacts,
                    }
                    local_checkpoint = LOCAL_CHECKPOINT_ROOT / namespace / f'{video_id}.json'
                    atomic_write_json(local_checkpoint, checkpoint)
                    if container:
                        client = container.get_blob_client(checkpoint_blob_name(namespace, video_id))
                        client.upload_blob(json.dumps(checkpoint, ensure_ascii=False).encode('utf-8'), overwrite=True,
                                           content_settings=ContentSettings(content_type='application/json'))
                    completed += 1
                    total_processed_frames += len(frames)
                    elapsed = time.perf_counter() - run_started
                    throughput = total_processed_frames / elapsed if elapsed else 0
                    remaining_frames = sum(
                        len(x.get('frames', x.get('frame_blobs', [])))
                        for x in pending_jobs[pending_index + 1:]
                    )
                    eta_hours = remaining_frames / throughput / 3600 if throughput else float('inf')
                    print(
                        f'DONE {video_id}: encode {encode_seconds:.1f}s '
                        f'({len(frames) / max(encode_seconds, 1e-6):.2f} frame/s), '
                        f'total {time.perf_counter() - video_started:.1f}s, ETA {eta_hours:.2f}h'
                    )
                except Exception as exc:
                    failed.append({'namespace': namespace, 'video_id': video_id, 'error': repr(exc)})
                    print('FAILED:', namespace, video_id, repr(exc))
                    if pending_index + 1 < len(pending_jobs) and next_download is None:
                        next_download = schedule_download(pending_jobs[pending_index + 1])
                finally:
                    cleanup_job_frames(namespace, video_id)
                    current_download = next_download
        finally:
            if prefetch_pool is not None:
                prefetch_pool.shutdown(wait=True)
            for model, _, _ in loaded_models.values():
                unload(model)

        report = {'part_name': PART_NAME, 'completed': completed, 'failed': failed, 'total': len(jobs)}
        report_path = WORK_ROOT / 'report.json'
        atomic_write_json(report_path, report)
        if container:
            upload_verified(report_path, f'reports/{EMBEDDING_RUN}/{PART_NAME}.json', 'application/json')
        print('\\nFINAL:', json.dumps(report, ensure_ascii=False, indent=2))
        if failed:
            raise RuntimeError(f'{len(failed)} video(s) failed; read report.json and rerun to resume.')
        """
    ),
    markdown(
        """
        ## Next step: merge, do not concatenate indexes from five workers

        Every worker uses independent per-video NPY files to remain resumable and disk-safe.
        After all five reports are complete, run the Jina merger which:

        1. downloads Jina files and records, assigns **new global contiguous vector IDs**,
           and writes `jina_faiss.index`, `global_ids.parquet`, `video_metadata.parquet`,
           and `index_meta.json`;
        2. validates FAISS `ntotal`, vector dimension, record count, duplicate keys, and
           timestamp coverage before publishing either index.

        Do not point the current BEiT3 retriever at Jina artifacts. Query text must be encoded
        by `jinaai/jina-clip-v2` with the same `JINA_TRUNCATE_DIM` before FAISS search.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUTPUT)


# A separate merger is required: part-local vector rows have no global FAISS IDs.
MERGER_OUTPUT = ROOT / "merge-azure-jina-embedding-index.ipynb"
merger_cells = [
    markdown(
        """
        # AIC 2026 - Merge Azure Jina embedding parts into a final FAISS index

        This notebook verifies completion directly from per-video checkpoints and the Azure
        keyframe directory tree. Worker reports are informative but are not required.
        The resulting Jina index requires a Jina text-query retriever in the backend.
        """
    ),
    code("""!pip install -q azure-storage-blob pyarrow faiss-cpu pandas"""),
    code(
        """
        from pathlib import Path
        import json, os
        import numpy as np
        import pandas as pd
        import faiss
        from azure.storage.blob import BlobServiceClient, ContentSettings

        EMBEDDING_RUN = 'fine_keyframes_jina_clip_v2_1024d_v2'
        AZURE_CONTAINER_EMBEDDINGS = 'embeddings'
        AZURE_CONTAINER_KEYFRAMES = 'keyframes'
        AZURE_KEYFRAMES_PREFIX = ''
        OUTPUT_ROOT = Path('/kaggle/working') / f'{EMBEDDING_RUN}_merged_indexes'
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        MODEL_NAMES = ['jina']

        def get_secret(name):
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return os.environ.get(name)

        connection_string = get_secret('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise RuntimeError('Missing Kaggle Secret AZURE_STORAGE_CONNECTION_STRING')
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container = blob_service.get_container_client(AZURE_CONTAINER_EMBEDDINGS)
        keyframes_container = blob_service.get_container_client(AZURE_CONTAINER_KEYFRAMES)
        container.get_container_properties()
        keyframes_container.get_container_properties()
        """
    ),
    code(
        """
        # Reports are useful progress summaries, but a Kaggle session can stop after the
        # last checkpoint and before writing its final report. Read them without requiring them.
        reports = []
        prefix = f'reports/{EMBEDDING_RUN}/'
        for blob in container.list_blobs(name_starts_with=prefix):
            reports.append(json.loads(container.get_blob_client(blob.name).download_blob().readall()))
        print('Reports found:', len(reports))
        for report in sorted(reports, key=lambda item: item['part_name']):
            print(report['part_name'], 'completed=', report['completed'], 'total=', report['total'], 'failed=', len(report['failed']))
        if not reports:
            print('No final reports found. Completion will be verified from keyframes + checkpoints.')
        elif any(report.get('failed') for report in reports):
            print('WARNING: at least one report contains failures; checkpoint verification below is authoritative.')
        """
    ),
    code(
        """
        # Stream checkpoints, records, and NPY blobs. Vector IDs are assigned here, once.
        import io
        from collections import defaultdict

        def download_json(blob_name):
            return json.loads(container.get_blob_client(blob_name).download_blob().readall())

        def checkpoints():
            prefix = f'checkpoints/{EMBEDDING_RUN}/'
            items = []
            for blob in container.list_blobs(name_starts_with=prefix):
                item = download_json(blob.name)
                if item.get('status') == 'completed':
                    items.append(item)
            return sorted(items, key=lambda x: (x['parent_namespace'] if 'parent_namespace' in x else x['namespace'], x['video_id']))

        items = checkpoints()
        keys = [(x['namespace'], x['video_id']) for x in items]
        if len(keys) != len(set(keys)):
            raise RuntimeError('Duplicate video checkpoint(s) found; clean the Azure run before merge.')
        print('Completed video checkpoints:', len(items))

        JOB_PRESETS = {
            'part_01': ['L21_a', 'L22_a', 'L23_a'],
            'part_02': ['L24_a', 'L25_a', 'L26_a'],
            'part_03': ['L26_b', 'L26_c', 'L26_d'],
            'part_04': ['L26_e', 'L27_a', 'L28_a'],
            'part_05': ['L29_a', 'L30_a'],
        }

        def namespace_prefix(namespace):
            return '/'.join(
                value for value in (AZURE_KEYFRAMES_PREFIX.strip('/'), namespace)
                if value
            ) + '/'

        expected_part_by_key = {}
        for part_name, namespaces in JOB_PRESETS.items():
            for namespace in namespaces:
                prefix = namespace_prefix(namespace)
                video_ids = set()
                # Hierarchical listing returns one virtual directory per video without
                # downloading/listing every individual keyframe blob.
                for entry in keyframes_container.walk_blobs(name_starts_with=prefix, delimiter='/'):
                    relative = entry.name[len(prefix):].strip('/')
                    if relative:
                        video_ids.add(relative.split('/')[0])
                print(part_name, namespace, 'expected videos=', len(video_ids))
                for video_id in video_ids:
                    key = (namespace, video_id)
                    if key in expected_part_by_key:
                        raise RuntimeError(f'Duplicate expected video assignment: {key}')
                    expected_part_by_key[key] = part_name

        if not expected_part_by_key:
            raise RuntimeError('No videos discovered in the Azure keyframes container; check AZURE_KEYFRAMES_PREFIX.')

        actual_by_key = {(item['namespace'], item['video_id']): item for item in items}
        expected_keys = set(expected_part_by_key)
        actual_keys = set(actual_by_key)
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        wrong_part = sorted(
            (key, expected_part_by_key[key], actual_by_key[key].get('part_name'))
            for key in expected_keys & actual_keys
            if actual_by_key[key].get('part_name') != expected_part_by_key[key]
        )

        print('Expected videos:', len(expected_keys))
        print('Checkpoint videos:', len(actual_keys))
        print('Missing:', len(missing_keys), '| Extra:', len(extra_keys), '| Wrong part:', len(wrong_part))
        if missing_keys:
            print('Missing sample:', missing_keys[:20])
        if extra_keys:
            print('Extra sample:', extra_keys[:20])
        if wrong_part:
            print('Wrong-part sample:', wrong_part[:20])
        if missing_keys or extra_keys or wrong_part:
            raise RuntimeError('Embedding checkpoints do not exactly cover the Azure keyframe videos. Rerun the affected workers.')

        print('PASS: every Azure video has exactly one completed checkpoint in the correct part.')
        """
    ),
    code(
        """
        def build_model_index(model_name):
            model_root = OUTPUT_ROOT / model_name
            model_root.mkdir(parents=True, exist_ok=True)
            index = None
            records, video_rows, vector_id = [], [], 0

            for item_no, item in enumerate(items, 1):
                artifact = item['artifacts'].get(model_name)
                if not artifact:
                    raise RuntimeError(f'{model_name} missing for {item["namespace"]}/{item["video_id"]}')
                record_payload = download_json(item['records_blob'])
                frame_records = record_payload['records']
                raw = container.get_blob_client(artifact['blob']).download_blob().readall()
                vectors = np.load(io.BytesIO(raw), allow_pickle=False).astype('float32', copy=False)
                if vectors.ndim != 2 or len(vectors) != len(frame_records) or not np.isfinite(vectors).all():
                    raise RuntimeError(f'Invalid {model_name} artifact: {artifact["blob"]}')
                norms = np.linalg.norm(vectors, axis=1)
                if not np.allclose(norms, 1.0, atol=2e-3):
                    raise RuntimeError(f'Non-normalized {model_name} artifact: {artifact["blob"]}')
                if index is None:
                    index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(vectors.shape[1])))
                elif index.d != vectors.shape[1]:
                    raise RuntimeError(f'{model_name} dimension changed within run')
                ids = np.arange(vector_id, vector_id + len(vectors), dtype=np.int64)
                index.add_with_ids(vectors, ids)
                for offset, record in enumerate(frame_records):
                    row = dict(record)
                    row['vector_id'] = int(vector_id + offset)
                    records.append(row)
                video_rows.append({
                    'video_id': item['video_id'], 'parent_namespace': item['namespace'],
                    'frame_count': len(vectors), 'embedding_dim': int(vectors.shape[1]),
                    'first_vector_id': int(vector_id), 'artifact_blob': artifact['blob'],
                })
                vector_id += len(vectors)
                if item_no % 25 == 0:
                    print(model_name, item_no, '/', len(items), 'videos;', vector_id, 'vectors')

            if index is None or index.ntotal != len(records):
                raise RuntimeError(f'{model_name}: index/record mismatch')
            global_ids = pd.DataFrame.from_records(records)
            if global_ids['vector_id'].tolist() != list(range(len(global_ids))):
                raise RuntimeError(f'{model_name}: non-contiguous vector IDs')
            global_ids.to_parquet(model_root / 'global_ids.parquet', index=False)
            pd.DataFrame.from_records(video_rows).to_parquet(model_root / 'video_metadata.parquet', index=False)
            faiss.write_index(index, str(model_root / f'{model_name}_faiss.index'))
            meta = {
                'embedding_run': EMBEDDING_RUN, 'model': model_name, 'embedding_dim': int(index.d),
                'metric': 'inner_product_on_l2_normalized_vectors', 'vector_count': int(index.ntotal),
                'video_count': len(video_rows), 'source': 'Azure per-video artifacts',
            }
            (model_root / 'index_meta.json').write_text(json.dumps(meta, indent=2) + '\\n', encoding='utf-8')
            print('Built', model_name, 'vectors=', index.ntotal, 'dim=', index.d)
            return model_root, meta

        built = [build_model_index(model_name) for model_name in MODEL_NAMES]
        """
    ),
    code(
        """
        # Validate locally, then publish final artifacts to Azure under a versioned prefix.
        import hashlib

        def sha256_path(path):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()

        for model_root, meta in built:
            model_name = meta['model']
            index_path = model_root / f'{model_name}_faiss.index'
            loaded_index = faiss.read_index(str(index_path))
            ids = pd.read_parquet(model_root / 'global_ids.parquet')
            assert loaded_index.ntotal == len(ids) == meta['vector_count']
            assert loaded_index.d == meta['embedding_dim']
            assert ids['vector_id'].is_unique
            assert ids['vector_id'].tolist() == list(range(len(ids)))
            assert ids[['parent_namespace', 'video_id', 'frame_path']].duplicated().sum() == 0
            print('PASS:', model_name, meta)

            for path in model_root.iterdir():
                remote = f'indexes/{EMBEDDING_RUN}/{model_name}/{path.name}'
                with path.open('rb') as handle:
                    container.get_blob_client(remote).upload_blob(
                        handle, overwrite=True, metadata={'sha256': sha256_path(path)},
                        content_settings=ContentSettings(content_type='application/octet-stream'),
                    )
                print('Uploaded:', remote)

        print('\\nJina artifacts for the matching backend retriever:')
        print('JINA_FAISS_INDEX_PATH=.../jina_faiss.index')
        print('JINA_GLOBAL_IDS_PATH=.../global_ids.parquet')
        print('JINA_VIDEO_METADATA_PATH=.../video_metadata.parquet')
        print('JINA_INDEX_META_PATH=.../index_meta.json')
        """
    ),
]

merger_notebook = {
    "cells": merger_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
MERGER_OUTPUT.write_text(json.dumps(merger_notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(MERGER_OUTPUT)
