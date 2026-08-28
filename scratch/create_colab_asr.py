import json
import os

def create_colab_asr_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎙️ AIC 2026 - Video/Audio ASR Extraction Pipeline (faster-whisper)\n",
                    "### 🚀 Google Colab GPU Edition - Trích xuất giọng nói cho video từ **L21 đến L30**\n",
                    "\n",
                    "Notebook này sử dụng mô hình **faster-whisper (Large-v3-Turbo)** chạy trên GPU Colab:\n",
                    "- **Tốc độ siêu nhanh**: Dùng CTranslate2 + FP16 trên T4 GPU (nhanh gấp 4–5 lần Whisper gốc).\n",
                    "- **Hỗ trợ file Zip & Google Drive**: Giải nén siêu tốc vào SSD `/content/` chỉ mất vài giây.\n",
                    "- **Tự động căn chỉnh Keyframe**: Map timestamp `start_time` và `end_time` với frame gần nhất (`nearest_faiss_id`, `nearest_frame_name`).\n",
                    "- **Đúng 100% Schema Backend**: Xuất ra `asr_results.json` nạp trực tiếp vào Elasticsearch `aic_asr`.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📦 1. Cài đặt các thư viện cần thiết"
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
                    "print(f\"CUDA Sẵn sàng: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU Thiết bị: {torch.cuda.get_device_name(0)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📂 2. Kết nối Google Drive & Giải nén Dữ liệu Video / Media (Nếu có)\n",
                    "*(Nếu bạn có file `Videos.zip` hoặc `map-keyframes.zip` trên Google Drive, chạy ô này để giải nén trực tiếp vào SSD Colab)*"
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
                    "\n",
                    "# Thư mục đích trên SSD của Colab\n",
                    "SSD_MEDIA_DIR = Path(\"/content/videos\")\n",
                    "SSD_MAP_DIR = Path(\"/content/map-keyframes\")\n",
                    "\n",
                    "# Tự động tìm và giải nén file zip nếu có trên Drive (Ví dụ: Videos.zip hoặc map-keyframes.zip)\n",
                    "zip_candidates = [\n",
                    "    \"/content/drive/MyDrive/Videos.zip\",\n",
                    "    \"/content/drive/MyDrive/AIC_Data/Videos.zip\",\n",
                    "    \"/content/drive/MyDrive/Videos_L21_L30.zip\"\n",
                    "]\n",
                    "\n",
                    "for zf in zip_candidates:\n",
                    "    if os.path.exists(zf):\n",
                    "        print(f\"Đang giải nén {zf} vào SSD Colab...\")\n",
                    "        !unzip -q \"{zf}\" -d /content/videos\n",
                    "        break\n",
                    "\n",
                    "# Giải nén map-keyframes nếu có\n",
                    "map_zip = \"/content/drive/MyDrive/map-keyframes.zip\"\n",
                    "if os.path.exists(map_zip):\n",
                    "    print(\"Đang giải nén map-keyframes.zip...\")\n",
                    "    !unzip -q \"{map_zip}\" -d /content/map-keyframes\n",
                    "\n",
                    "print(\"✅ Bước chuẩn bị dữ liệu đã xong!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 3. Cấu hình Đường dẫn & Tìm kiếm Danh sách Video `L21` -> `L30`"
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
                    "\n",
                    "OUTPUT_JSON = Path(\"/content/asr_results.json\")\n",
                    "CHECKPOINT_JSON = Path(\"/content/asr_results_checkpoint.json\")\n",
                    "\n",
                    "# Tìm thư mục chứa video trên Colab hoặc Drive\n",
                    "MEDIA_DIR = None\n",
                    "candidate_dirs = [\n",
                    "    Path(\"/content/videos\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC_Data/Videos\"),\n",
                    "    Path(\"/content/drive/MyDrive/Videos\"),\n",
                    "    Path(\"/content\")\n",
                    "]\n",
                    "\n",
                    "for cand in candidate_dirs:\n",
                    "    if cand.exists():\n",
                    "        # Kiểm tra xem có video không\n",
                    "        found_v = any(f.lower().endswith(('.mp4', '.mkv', '.avi', '.mp3', '.wav', '.m4a')) for _, _, files in os.walk(cand) for f in files[:10])\n",
                    "        if found_v:\n",
                    "            MEDIA_DIR = cand\n",
                    "            break\n",
                    "\n",
                    "print(f\"📁 Thư mục Video tìm thấy: {MEDIA_DIR}\")\n",
                    "\n",
                    "# Quét danh sách video từ L21 đến L30\n",
                    "valid_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.flac'}\n",
                    "all_files = []\n",
                    "if MEDIA_DIR and MEDIA_DIR.exists():\n",
                    "    for root, _, files in os.walk(MEDIA_DIR):\n",
                    "        for f in files:\n",
                    "            if os.path.splitext(f)[1].lower() in valid_exts:\n",
                    "                all_files.append(os.path.join(root, f))\n",
                    "\n",
                    "# Lọc video thuộc dải L21 đến L30\n",
                    "media_files = []\n",
                    "for vf in all_files:\n",
                    "    basename = os.path.basename(vf)\n",
                    "    if re.search(r\"L(2[1-9]|30)_V\\d+\", basename):\n",
                    "        media_files.append(vf)\n",
                    "\n",
                    "if not media_files:\n",
                    "    media_files = sorted(all_files) # fallback\n",
                    "else:\n",
                    "    media_files = sorted(media_files)\n",
                    "\n",
                    "print(f\"🔥 Tìm thấy {len(media_files)} video/audio (L21 -> L30) để xử lý ASR.\")"
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
                    "    # Tìm các thư mục map-keyframes\n",
                    "    search_roots = [Path(\"/content/map-keyframes\"), Path(\"/content/drive/MyDrive/AIC/Backend/Back-End/src/dict/map-keyframes\"), Path(\"/content\")]\n",
                    "    csv_files = []\n",
                    "    for s_root in search_roots:\n",
                    "        if s_root.exists():\n",
                    "            for root, _, files in os.walk(s_root):\n",
                    "                for f in files:\n",
                    "                    if f.lower().endswith('.csv'):\n",
                    "                        csv_files.append(os.path.join(root, f))\n",
                    "            if csv_files:\n",
                    "                break\n",
                    "                \n",
                    "    for csv_f in csv_files:\n",
                    "        v_id = os.path.splitext(os.path.basename(csv_f))[0]\n",
                    "        if not re.search(r\"L(2[1-9]|30)_V\\d+\", v_id):\n",
                    "            continue\n",
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
                    "        print(f\"✅ Đã nạp keyframes map cho {len(video_keyframe_map)} videos từ CSVs.\")\n",
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
                    "## 🤖 5. Khởi tạo Mô hình Whisper trên GPU Colab (`large-v3-turbo`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from faster_whisper import WhisperModel\n",
                    "\n",
                    "MODEL_SIZE = \"large-v3-turbo\"\n",
                    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                    "COMPUTE_TYPE = \"float16\" if torch.cuda.is_available() else \"int8\"\n",
                    "\n",
                    "print(f\"Đang nạp faster-whisper [{MODEL_SIZE}] lên GPU Colab ({COMPUTE_TYPE})...\")\n",
                    "model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)\n",
                    "print(\"✅ Whisper model đã sẵn sàng trên GPU!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 6. KÍCH HOẠT GPU: Thực hiện Trích xuất ASR Toàn bộ Video L21 -> L30"
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
                    "# Nạp checkpoint nếu có\n",
                    "processed_videos = set()\n",
                    "asr_results = []\n",
                    "if CHECKPOINT_JSON.exists():\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            asr_results = json.load(f)\n",
                    "            processed_videos = {item[\"video_id\"] for item in asr_results}\n",
                    "        print(f\"🔄 Đã nạp {len(asr_results)} đoạn ASR từ checkpoint ({len(processed_videos)} videos).\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Lỗi nạp checkpoint:\", e)\n",
                    "\n",
                    "for media_path in tqdm(media_files, desc=\"🚀 GPU Trích xuất ASR L21 -> L30\"):\n",
                    "    video_id = os.path.splitext(os.path.basename(media_path))[0]\n",
                    "    m = re.search(r\"(L\\d+_V\\d+)\", video_id)\n",
                    "    if m:\n",
                    "        video_id = m.group(1)\n",
                    "        \n",
                    "    if video_id in processed_videos:\n",
                    "        continue\n",
                    "        \n",
                    "    try:\n",
                    "        segments, info = model.transcribe(\n",
                    "            media_path,\n",
                    "            language=\"vi\",\n",
                    "            beam_size=5,\n",
                    "            word_timestamps=False,\n",
                    "            vad_filter=True, # Bỏ qua đoạn im lặng\n",
                    "            vad_parameters=dict(min_silence_duration_ms=500)\n",
                    "        )\n",
                    "        \n",
                    "        video_segments = []\n",
                    "        for seg in segments:\n",
                    "            start_time = round(float(seg.start), 3)\n",
                    "            end_time = round(float(seg.end), 3)\n",
                    "            text = seg.text.strip()\n",
                    "            if not text:\n",
                    "                continue\n",
                    "                \n",
                    "            mid_time = (start_time + end_time) / 2.0\n",
                    "            nearest_frame = find_nearest_keyframe(mid_time, video_id, keyframe_index)\n",
                    "            \n",
                    "            doc = {\n",
                    "                \"video_id\": video_id,\n",
                    "                \"start_time\": start_time,\n",
                    "                \"end_time\": end_time,\n",
                    "                \"text\": text,\n",
                    "                \"nearest_faiss_id\": int(nearest_frame[\"faiss_id\"]),\n",
                    "                \"nearest_frame_name\": str(nearest_frame[\"frame_name\"])\n",
                    "            }\n",
                    "            video_segments.append(doc)\n",
                    "            \n",
                    "        asr_results.extend(video_segments)\n",
                    "        processed_videos.add(video_id)\n",
                    "        \n",
                    "        # Tự động lưu checkpoint sau mỗi video\n",
                    "        with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"Lỗi xử lý {media_path}: {e}\")\n",
                    "\n",
                    "# Lưu kết quả chính thức\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "# Nếu Google Drive được mount, lưu thêm 1 bản dự phòng vào Drive\n",
                    "drive_save = Path(\"/content/drive/MyDrive/asr_results.json\")\n",
                    "try:\n",
                    "    with open(drive_save, 'w', encoding='utf-8') as f:\n",
                    "        json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "    print(f\"💾 Đã tự động lưu 1 bản dự phòng vào Google Drive: {drive_save}\")\n",
                    "except Exception:\n",
                    "    pass\n",
                    "\n",
                    "print(f\"\\n🎉 HOÀN TẤT TOÀN BỘ! Đã lưu {len(asr_results)} đoạn ASR vào: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 7. Tải file `asr_results.json` về máy tính"
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
                    "    print(\"\\n👉 Đang tải file về trình duyệt của bạn...\")\n",
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
colab_asr_nb = create_colab_asr_notebook()

# Write to scripts/notebooks/extract_asr_colab.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "extract_asr_colab.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_asr_nb, f, indent=1, ensure_ascii=False)

# Write to scripts/notebooks/04_Extract_ASR_Whisper.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "04_Extract_ASR_Whisper.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_asr_nb, f, indent=1, ensure_ascii=False)

print("Created extract_asr_colab.ipynb specifically optimized for Google Colab GPU!")
