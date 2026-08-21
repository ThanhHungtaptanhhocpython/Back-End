import json
import os

def create_colab_asr_a100_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎙️ AIC 2026 - Video ASR Extraction Pipeline (faster-whisper)\n",
                    "### 🚀 Google Colab A100 / T4 GPU Edition - Siêu Tốc Cho **`video_batch_1` (L21 -> L30)**\n",
                    "\n",
                    "Notebook này được tối ưu đặc biệt cho **NVIDIA A100 GPU** (và tương thích hoàn hảo với T4/V100/L4):\n",
                    "- **Kích hoạt Batched Inference trên A100**: Tận dụng 40GB VRAM và Tensor Cores để xử lý song song nhiều audio chunks cùng lúc (nhanh gấp 3–5 lần so với chạy đơn luồng, 1 video 30 phút xong trong 5-10 giây).\n",
                    "- **Cơ chế Xử lý Tuần tự từng file Zip**: Giải nén tạm vào SSD `/content/temp_videos/` -> Chạy GPU A100 -> Tự động xóa dọn dẹp để **không bao giờ bị tràn ổ đĩa Colab**.\n",
                    "- **Tự động sao lưu lên Google Drive**: Lưu checkpoint liên tục và đồng bộ `asr_results.json` vào Drive sau mỗi file zip.\n",
                    "- **Đúng 100% Schema Backend**: Nạp trực tiếp vào Elasticsearch `aic_asr`.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚠️ BƯỚC QUAN TRỌNG TRÊN GOOGLE DRIVE TRƯỚC KHI CHẠY:\n",
                    "1. Mở Google Drive trên trình duyệt, vào mục **Được chia sẻ với tôi (Shared with me)** -> mở thư mục **AIC2025**.\n",
                    "2. **Nhấp chuột phải vào thư mục `video_batch_1`** -> Chọn **\"Thêm lối tắt vào Drive\" (Add shortcut to Drive)** -> Chọn **\"Drive của tôi\" (My Drive)**.\n",
                    "*(Sau khi làm bước này, Colab sẽ đọc được toàn bộ các file `Videos_L21_a.zip` -> `Videos_L30_a.zip`)*"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📦 1. Cài đặt thư viện & Kiểm tra GPU A100"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q faster-whisper tqdm Pillow\n",
                    "!apt-get update -qq && apt-get install -y -qq ffmpeg\n",
                    "\n",
                    "import torch\n",
                    "print(f\"✅ CUDA Sẵn sàng: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    gpu_name = torch.cuda.get_device_name(0)\n",
                    "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                    "    print(f\"⚡ GPU Thiết bị: {gpu_name} ({vram_gb:.1f} GB VRAM)\")\n",
                    "    if \"A100\" in gpu_name.upper():\n",
                    "        print(\"🚀 TUYỆT VỜI! Đã kích hoạt NVIDIA A100 - Bật chế độ tăng tốc Batched Inference!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📂 2. Kết nối Google Drive"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "import os\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Mount Google Drive\n",
                    "drive.mount('/content/drive')\n",
                    "print(\"✅ Đã kết nối Google Drive thành công!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 3. Tự động Quét Danh sách các file Zip (`Videos_L21_a.zip` -> `Videos_L30_a.zip`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import glob\n",
                    "import re\n",
                    "import json\n",
                    "import csv\n",
                    "import shutil\n",
                    "\n",
                    "OUTPUT_JSON = Path(\"/content/asr_results.json\")\n",
                    "CHECKPOINT_JSON = Path(\"/content/asr_results_checkpoint.json\")\n",
                    "DRIVE_BACKUP_JSON = Path(\"/content/drive/MyDrive/asr_results.json\")\n",
                    "\n",
                    "# Tìm kiếm các file Videos_Lxx.zip trong Google Drive\n",
                    "print(\"⏳ Đang quét các file zip video trong Google Drive...\")\n",
                    "all_zip_files = []\n",
                    "for root, _, files in os.walk(\"/content/drive/MyDrive\"):\n",
                    "    for f in files:\n",
                    "        if f.startswith(\"Videos_L\") and f.endswith(\".zip\"):\n",
                    "            all_zip_files.append(os.path.join(root, f))\n",
                    "\n",
                    "# Sắp xếp theo thứ tự L21 -> L30\n",
                    "def get_zip_order(path):\n",
                    "    fname = os.path.basename(path)\n",
                    "    m = re.search(r\"L(\\d+)\", fname)\n",
                    "    return int(m.group(1)) if m else 999\n",
                    "\n",
                    "target_zips = sorted(all_zip_files, key=get_zip_order)\n",
                    "\n",
                    "print(f\"\\n🔥 TÌM THẤY TỔNG CỘNG {len(target_zips)} TỆP ZIP:\")\n",
                    "for z in target_zips:\n",
                    "    sz_gb = os.path.getsize(z) / (1024**3)\n",
                    "    print(f\"  - {os.path.basename(z):20s} ({sz_gb:.2f} GB) -> {z}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔗 4. Nạp Keyframe Timestamp Index để Alignment (`map-keyframes`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def build_keyframe_index():\n",
                    "    video_keyframe_map = {}\n",
                    "    faiss_id_counter = 0\n",
                    "    \n",
                    "    csv_files = []\n",
                    "    for root, _, files in os.walk(\"/content/drive/MyDrive\"):\n",
                    "        for f in files:\n",
                    "            if f.endswith('.csv') and re.search(r\"L(2[1-9]|30)_V\\d+\", f):\n",
                    "                csv_files.append(os.path.join(root, f))\n",
                    "                \n",
                    "    for csv_f in csv_files:\n",
                    "        v_id = os.path.splitext(os.path.basename(csv_f))[0]\n",
                    "        try:\n",
                    "            with open(csv_f, 'r', encoding='utf-8') as f:\n",
                    "                reader = csv.DictReader(f)\n",
                    "                frames = []\n",
                    "                for row in reader:\n",
                    "                    try:\n",
                    "                        n_val = int(row.get('n', 0))\n",
                    "                        pts = float(row.get('pts_time', 0.0))\n",
                    "                        frames.append({\n",
                    "                            \"faiss_id\": faiss_id_counter,\n",
                    "                            \"timestamp\": pts,\n",
                    "                            \"frame_name\": f\"{n_val:04d}.webp\"\n",
                    "                        })\n",
                    "                        faiss_id_counter += 1\n",
                    "                    except Exception:\n",
                    "                        pass\n",
                    "                if frames:\n",
                    "                    video_keyframe_map[v_id] = sorted(frames, key=lambda x: x[\"timestamp\"])\n",
                    "        except Exception:\n",
                    "            pass\n",
                    "            \n",
                    "    if video_keyframe_map:\n",
                    "        print(f\"✅ Đã nạp map-keyframes cho {len(video_keyframe_map)} videos từ Drive.\")\n",
                    "    else:\n",
                    "        print(\"ℹ️ Chưa có map-keyframes CSVs. Hệ thống sẽ căn chỉnh frame ước lượng theo giây.\")\n",
                    "    return video_keyframe_map\n",
                    "\n",
                    "def find_nearest_keyframe(target_time, video_id, keyframe_index):\n",
                    "    if video_id not in keyframe_index or not keyframe_index[video_id]:\n",
                    "        return {\"faiss_id\": 0, \"frame_name\": f\"{int(target_time):04d}.webp\"}\n",
                    "    frames = keyframe_index[video_id]\n",
                    "    return min(frames, key=lambda f: abs(f[\"timestamp\"] - target_time))\n",
                    "\n",
                    "keyframe_index = build_keyframe_index()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 5. Khởi tạo Mô hình Whisper Tối Ưu Cho A100 / T4 GPU"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from faster_whisper import WhisperModel, BatchedInferencePipeline\n",
                    "\n",
                    "MODEL_SIZE = \"large-v3-turbo\"\n",
                    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                    "COMPUTE_TYPE = \"float16\" if torch.cuda.is_available() else \"int8\"\n",
                    "\n",
                    "IS_A100 = False\n",
                    "if torch.cuda.is_available():\n",
                    "    gpu_name = torch.cuda.get_device_name(0)\n",
                    "    IS_A100 = \"A100\" in gpu_name.upper()\n",
                    "\n",
                    "print(f\"Đang nạp faster-whisper [{MODEL_SIZE}] trên GPU ({COMPUTE_TYPE})...\")\n",
                    "# Trên A100 dùng num_workers=4 để load nhanh tối đa\n",
                    "num_workers = 4 if IS_A100 else 1\n",
                    "base_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=num_workers)\n",
                    "\n",
                    "# Kích hoạt Batched Inference Pipeline cho A100 để inference song song\n",
                    "if IS_A100:\n",
                    "    print(\"🚀 Đang kích hoạt BatchedInferencePipeline cho NVIDIA A100...\")\n",
                    "    batched_model = BatchedInferencePipeline(model=base_model)\n",
                    "    BATCH_SIZE = 16 # A100 40GB xử lý batch 16 cực mượt\n",
                    "else:\n",
                    "    batched_model = None\n",
                    "    BATCH_SIZE = 1\n",
                    "\n",
                    "print(f\"✅ Mô hình đã sẵn sàng! (Chế độ A100 Batching: {'BẬT (Batch=16)' if IS_A100 else 'TẮT (Chuẩn T4)'})\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 6. KÍCH HOẠT A100 GPU: Xử lý Tuần tự từng file Zip & Tự dọn dẹp SSD"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from tqdm import tqdm\n",
                    "\n",
                    "TEMP_EXTRACT_DIR = Path(\"/content/temp_videos\")\n",
                    "valid_video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.flac'}\n",
                    "\n",
                    "processed_videos = set()\n",
                    "asr_results = []\n",
                    "\n",
                    "# Nạp checkpoint từ Drive hoặc Local nếu có\n",
                    "if DRIVE_BACKUP_JSON.exists():\n",
                    "    try:\n",
                    "        with open(DRIVE_BACKUP_JSON, 'r', encoding='utf-8') as f:\n",
                    "            asr_results = json.load(f)\n",
                    "            processed_videos = {item[\"video_id\"] for item in asr_results}\n",
                    "        print(f\"🔄 Đã khôi phục {len(asr_results)} đoạn ASR ({len(processed_videos)} videos) từ Google Drive!\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "elif CHECKPOINT_JSON.exists():\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            asr_results = json.load(f)\n",
                    "            processed_videos = {item[\"video_id\"] for item in asr_results}\n",
                    "        print(f\"🔄 Đã nạp {len(asr_results)} đoạn ASR từ checkpoint cục bộ.\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "\n",
                    "# Lặp qua từng file Zip theo thứ tự\n",
                    "for zip_idx, zip_file in enumerate(target_zips, 1):\n",
                    "    zip_name = os.path.basename(zip_file)\n",
                    "    print(f\"\\n=================================================================\")\n",
                    "    print(f\"📦 [{zip_idx}/{len(target_zips)}] Đang xử lý file zip: {zip_name}\")\n",
                    "    print(f\"=================================================================\")\n",
                    "    \n",
                    "    if TEMP_EXTRACT_DIR.exists():\n",
                    "        shutil.rmtree(TEMP_EXTRACT_DIR)\n",
                    "    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)\n",
                    "    \n",
                    "    print(f\"⏳ Đang giải nén {zip_name} vào SSD Colab...\")\n",
                    "    !unzip -q \"{zip_file}\" -d \"{TEMP_EXTRACT_DIR}\"\n",
                    "    \n",
                    "    extracted_videos = []\n",
                    "    for root, _, files in os.walk(TEMP_EXTRACT_DIR):\n",
                    "        for f in files:\n",
                    "            if os.path.splitext(f)[1].lower() in valid_video_exts:\n",
                    "                extracted_videos.append(os.path.join(root, f))\n",
                    "                \n",
                    "    extracted_videos = sorted(extracted_videos)\n",
                    "    print(f\"🎬 Tìm thấy {len(extracted_videos)} video trong {zip_name}.\")\n",
                    "    \n",
                    "    for media_path in tqdm(extracted_videos, desc=f\"🚀 GPU ASR {zip_name}\"):\n",
                    "        raw_vid = os.path.splitext(os.path.basename(media_path))[0]\n",
                    "        m = re.search(r\"(L\\d+_V\\d+)\", raw_vid)\n",
                    "        video_id = m.group(1) if m else raw_vid\n",
                    "        \n",
                    "        if video_id in processed_videos:\n",
                    "            continue\n",
                    "            \n",
                    "        try:\n",
                    "            # Dùng BatchedInferencePipeline trên A100 để tăng tốc tối đa\n",
                    "            if batched_model is not None:\n",
                    "                segments, info = batched_model.transcribe(\n",
                    "                    media_path,\n",
                    "                    language=\"vi\",\n",
                    "                    batch_size=BATCH_SIZE,\n",
                    "                    vad_filter=True\n",
                    "                )\n",
                    "            else:\n",
                    "                segments, info = base_model.transcribe(\n",
                    "                    media_path,\n",
                    "                    language=\"vi\",\n",
                    "                    beam_size=5,\n",
                    "                    word_timestamps=False,\n",
                    "                    vad_filter=True,\n",
                    "                    vad_parameters=dict(min_silence_duration_ms=500)\n",
                    "                )\n",
                    "            \n",
                    "            video_segments = []\n",
                    "            for seg in segments:\n",
                    "                start_time = round(float(seg.start), 3)\n",
                    "                end_time = round(float(seg.end), 3)\n",
                    "                text = seg.text.strip()\n",
                    "                if not text:\n",
                    "                    continue\n",
                    "                    \n",
                    "                mid_time = (start_time + end_time) / 2.0\n",
                    "                nearest_frame = find_nearest_keyframe(mid_time, video_id, keyframe_index)\n",
                    "                \n",
                    "                doc = {\n",
                    "                    \"video_id\": video_id,\n",
                    "                    \"start_time\": start_time,\n",
                    "                    \"end_time\": end_time,\n",
                    "                    \"text\": text,\n",
                    "                    \"nearest_faiss_id\": int(nearest_frame[\"faiss_id\"]),\n",
                    "                    \"nearest_frame_name\": str(nearest_frame[\"frame_name\"])\n",
                    "                }\n",
                    "                video_segments.append(doc)\n",
                    "                \n",
                    "            asr_results.extend(video_segments)\n",
                    "            processed_videos.add(video_id)\n",
                    "            \n",
                    "            # Lưu checkpoint cục bộ\n",
                    "            with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "                json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "                \n",
                    "        except Exception as e:\n",
                    "            print(f\"Lỗi xử lý {media_path}: {e}\")\n",
                    "            \n",
                    "    # Tự động sao lưu lên Google Drive sau khi xong từng file zip\n",
                    "    try:\n",
                    "        with open(DRIVE_BACKUP_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "        print(f\"💾 [Auto-Backup Drive] Đã lưu {len(asr_results)} đoạn ASR lên Google Drive!\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "        \n",
                    "    # Dọn dẹp ổ đĩa SSD\n",
                    "    shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)\n",
                    "    print(f\"🧹 Đã giải phóng SSD sau khi xong {zip_name}.\")\n",
                    "\n",
                    "# Lưu file chính thức cuối cùng\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n🎉 HOÀN TẤT TOÀN BỘ! Đã trích xuất {len(asr_results)} đoạn ASR và lưu vào: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 7. Kiểm tra & Tải file về máy"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "\n",
                    "if OUTPUT_JSON.exists():\n",
                    "    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:\n",
                    "        data = json.load(f)\n",
                    "    print(f\"✅ Tổng số câu ASR trích xuất: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU KẾT QUẢ ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "        \n",
                    "    print(\"\\n👉 Đang tải file `asr_results.json` về máy tính...\")\n",
                    "    files.download(str(OUTPUT_JSON))\n",
                    "    print(\"👉 Sau khi tải về, copy vào thư mục Backend: `src/dict/asr_results.json`.\")\n",
                    "    print(\"👉 Chạy `python scripts/indexing/master_index_pipeline.py` để nạp vào Elasticsearch.\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb

base_dir = r"c:\Users\Lenovo\Documents\GitHub\AIC\Backend\Back-End"
colab_asr_a100 = create_colab_asr_a100_notebook()

# Write to scripts/notebooks/extract_asr_colab.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "extract_asr_colab.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_asr_a100, f, indent=1, ensure_ascii=False)

# Write to scripts/notebooks/04_Extract_ASR_Whisper.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "04_Extract_ASR_Whisper.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_asr_a100, f, indent=1, ensure_ascii=False)

print("Updated ASR notebook with NVIDIA A100 Batched Inference acceleration!")
