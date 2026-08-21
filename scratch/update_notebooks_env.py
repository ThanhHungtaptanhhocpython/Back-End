import json
import os

def create_ocr_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📌 AIC 2026 - Keyframe OCR Extraction Pipeline (PaddleOCR)\n",
                    "### 🚀 Tương thích: Google Colab GPU / VS Code Remote Colab / Local / Kaggle\n",
                    "\n",
                    "Notebook này tự động nạp cấu hình đường dẫn từ file **`.env`** của dự án (như `KEYFRAMES_ROOT`, `METADATA_PATH`), trích xuất text tiếng Việt bằng **PaddleOCR**, và lưu trực tiếp file **`src/dict/ocr_results.json`** chuẩn schema Elasticsearch `aic_ocr` của Backend.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Cài đặt các thư viện cần thiết\n",
                    "# Cố định numpy < 2.0 để tránh xung đột ABI với PaddlePaddle\n",
                    "!pip install \"numpy<2.0.0\" paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 python-dotenv Pillow tqdm pandas"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Tự động Mount Google Drive nếu chạy trực tiếp trên Colab\n",
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "IN_COLAB = False\n",
                    "try:\n",
                    "    from google.colab import drive\n",
                    "    drive.mount('/content/drive')\n",
                    "    IN_COLAB = True\n",
                    "    print(\"✅ Đã kết nối Google Drive thành công!\")\n",
                    "except ImportError:\n",
                    "    print(\"ℹ️ Đang chạy trên môi trường Local / Remote SSH (không cần Colab Drive mount).\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 3. Tự động đọc cấu hình đường dẫn từ `.env`\n",
                    "Tự động tìm kiếm file `.env` của Backend và nạp các biến `KEYFRAMES_ROOT`, `METADATA_PATH`..."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import glob\n",
                    "import json\n",
                    "import logging\n",
                    "import pandas as pd\n",
                    "from tqdm import tqdm\n",
                    "from PIL import Image\n",
                    "from dotenv import load_dotenv\n",
                    "\n",
                    "# Tìm thư mục gốc dự án Backend\n",
                    "def find_project_root():\n",
                    "    # Tìm từ thư mục hiện tại lên trên\n",
                    "    curr = Path(os.getcwd()).resolve()\n",
                    "    for parent in [curr] + list(curr.parents):\n",
                    "        if (parent / \".env\").exists() or (parent / \"src\" / \"dict\").exists():\n",
                    "            return parent\n",
                    "    # Nếu trên Colab và có liên kết thư mục trên Drive\n",
                    "    colab_drive_root = Path(\"/content/drive/MyDrive/AIC/Backend/Back-End\")\n",
                    "    if colab_drive_root.exists():\n",
                    "        return colab_drive_root\n",
                    "    return curr\n",
                    "\n",
                    "PROJECT_ROOT = find_project_root()\n",
                    "env_path = PROJECT_ROOT / \".env\"\n",
                    "\n",
                    "if env_path.exists():\n",
                    "    load_dotenv(dotenv_path=env_path)\n",
                    "    print(f\"✅ Đã load .env từ: {env_path}\")\n",
                    "else:\n",
                    "    print(f\"⚠️ Không tìm thấy file .env tại {env_path}, sử dụng cấu hình mặc định.\")\n",
                    "\n",
                    "# Lấy đường dẫn từ .env hoặc fallback\n",
                    "ENV_KEYFRAMES_ROOT = os.getenv(\"KEYFRAMES_ROOT\", \"\")\n",
                    "ENV_METADATA_PATH = os.getenv(\"METADATA_PATH\", \"\")\n",
                    "\n",
                    "# Thiết lập các đường dẫn chuẩn theo backend\n",
                    "DICT_DIR = PROJECT_ROOT / \"src\" / \"dict\"\n",
                    "OUTPUT_JSON = DICT_DIR / \"ocr_results.json\"\n",
                    "CHECKPOINT_JSON = DICT_DIR / \"ocr_results_checkpoint.json\"\n",
                    "MAP_KEYFRAMES_DIR = DICT_DIR / \"map-keyframes\"\n",
                    "\n",
                    "# Xác định đường dẫn Keyframes thực tế\n",
                    "KEYFRAMES_DIR = None\n",
                    "candidate_keyframes = [\n",
                    "    Path(ENV_KEYFRAMES_ROOT) if ENV_KEYFRAMES_ROOT else None,\n",
                    "    PROJECT_ROOT / \"src\" / \"data\" / \"Keyframes\",\n",
                    "    Path(\"D:/AIC_Data/Keyframes/keyframes_L21_onwards\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC_Data/Keyframes/keyframes_L21_onwards\"),\n",
                    "    Path(\"/content/drive/MyDrive/Keyframes\"),\n",
                    "    Path(\"/kaggle/input/your-keyframes-dataset/\"),\n",
                    "]\n",
                    "for cand in candidate_keyframes:\n",
                    "    if cand and cand.exists():\n",
                    "        KEYFRAMES_DIR = cand\n",
                    "        break\n",
                    "\n",
                    "# Xác định file metadata\n",
                    "METADATA_PATH = None\n",
                    "candidate_meta = [\n",
                    "    Path(ENV_METADATA_PATH) if ENV_METADATA_PATH else None,\n",
                    "    DICT_DIR / \"metadata_clip.json\",\n",
                    "    Path(\"/content/drive/MyDrive/metadata_clip.json\"),\n",
                    "    Path(\"/kaggle/input/your-metadata-dataset/metadata_clip.json\")\n",
                    "]\n",
                    "for cand in candidate_meta:\n",
                    "    if cand and cand.exists():\n",
                    "        METADATA_PATH = cand\n",
                    "        break\n",
                    "\n",
                    "print(f\"\\n📁 CẤU HÌNH ĐƯỜNG DẪN HOẠT ĐỘNG:\")\n",
                    "print(f\"- Project Root:     {PROJECT_ROOT}\")\n",
                    "print(f\"- Keyframes Dir:    {KEYFRAMES_DIR}\")\n",
                    "print(f\"- Metadata Path:    {METADATA_PATH}\")\n",
                    "print(f\"- Output Path:      {OUTPUT_JSON}\")\n",
                    "print(f\"- Map Keyframes:    {MAP_KEYFRAMES_DIR}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 4. Xây dựng Danh sách Keyframes cần trích xuất"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def build_keyframe_tasks(keyframes_dir, metadata_path=None, map_keyframes_dir=None):\n",
                    "    tasks = []\n",
                    "    if not keyframes_dir or not os.path.exists(keyframes_dir):\n",
                    "        print(f\"❌ Thư mục keyframes không tồn tại: {keyframes_dir}\")\n",
                    "        return tasks\n",
                    "\n",
                    "    # Cách 1: Nạp từ metadata_clip.json nếu có\n",
                    "    if metadata_path and os.path.exists(metadata_path):\n",
                    "        print(f\"Đang nạp từ metadata: {metadata_path}\")\n",
                    "        with open(metadata_path, 'r', encoding='utf-8') as f:\n",
                    "            meta = json.load(f)\n",
                    "            \n",
                    "        for fid_str, info in meta.items():\n",
                    "            faiss_id = int(fid_str)\n",
                    "            split = str(info.get(\"split\", \"\"))\n",
                    "            video_id = str(info.get(\"video_id\", \"\"))\n",
                    "            frame_name = str(info.get(\"frame_name\", \"\"))\n",
                    "            \n",
                    "            candidates = [\n",
                    "                Path(keyframes_dir) / split / video_id / frame_name,\n",
                    "                Path(keyframes_dir) / video_id / frame_name,\n",
                    "                Path(keyframes_dir) / frame_name,\n",
                    "                Path(keyframes_dir) / f\"Keyframes_{split}\" / video_id / frame_name,\n",
                    "            ]\n",
                    "            img_path = None\n",
                    "            for cand in candidates:\n",
                    "                if cand.exists():\n",
                    "                    img_path = str(cand)\n",
                    "                    break\n",
                    "                    \n",
                    "            if img_path:\n",
                    "                tasks.append({\n",
                    "                    \"image_path\": img_path,\n",
                    "                    \"faiss_id\": faiss_id,\n",
                    "                    \"video_id\": video_id,\n",
                    "                    \"frame_name\": frame_name,\n",
                    "                    \"split\": split,\n",
                    "                    \"global_frame_id\": int(info.get(\"global_frame_id\", faiss_id)),\n",
                    "                    \"timestamp\": float(info.get(\"timestamp\", 0.0))\n",
                    "                })\n",
                    "        print(f\"-> Khớp được {len(tasks)} ảnh trên ổ đĩa từ metadata.\")\n",
                    "        if tasks:\n",
                    "            return tasks\n",
                    "\n",
                    "    # Cách 2: Quét thư mục trực tiếp\n",
                    "    print(f\"Quét các file ảnh từ: {keyframes_dir}\")\n",
                    "    all_images = sorted(\n",
                    "        glob.glob(f\"{keyframes_dir}/**/*.webp\", recursive=True) +\n",
                    "        glob.glob(f\"{keyframes_dir}/**/*.jpg\", recursive=True) +\n",
                    "        glob.glob(f\"{keyframes_dir}/**/*.png\", recursive=True)\n",
                    "    )\n",
                    "    print(f\"Tìm thấy {len(all_images)} ảnh.\")\n",
                    "    \n",
                    "    csv_cache = {}\n",
                    "    if map_keyframes_dir and os.path.exists(map_keyframes_dir):\n",
                    "        for csv_file in glob.glob(f\"{map_keyframes_dir}/**/*.csv\", recursive=True):\n",
                    "            v_name = os.path.splitext(os.path.basename(csv_file))[0]\n",
                    "            try:\n",
                    "                csv_cache[v_name] = pd.read_csv(csv_file)\n",
                    "            except Exception:\n",
                    "                pass\n",
                    "\n",
                    "    faiss_id_counter = 0\n",
                    "    for img_p in all_images:\n",
                    "        p = Path(img_p)\n",
                    "        frame_name = p.name\n",
                    "        video_id = p.parent.name\n",
                    "        split = p.parent.parent.name if len(p.parts) > 2 else \"\"\n",
                    "        if \"Keyframes_\" in split:\n",
                    "            split = split.replace(\"Keyframes_\", \"\")\n",
                    "            \n",
                    "        timestamp = 0.0\n",
                    "        global_frame_id = faiss_id_counter\n",
                    "        \n",
                    "        if video_id in csv_cache:\n",
                    "            df = csv_cache[video_id]\n",
                    "            try:\n",
                    "                frame_num = int(os.path.splitext(frame_name)[0])\n",
                    "                row = df[df['n'] == frame_num]\n",
                    "                if not row.empty:\n",
                    "                    timestamp = float(row.iloc[0]['pts_time'])\n",
                    "                    global_frame_id = int(row.iloc[0].get('frame_idx', frame_num))\n",
                    "            except Exception:\n",
                    "                pass\n",
                    "                \n",
                    "        tasks.append({\n",
                    "            \"image_path\": img_p,\n",
                    "            \"faiss_id\": faiss_id_counter,\n",
                    "            \"video_id\": video_id,\n",
                    "            \"frame_name\": frame_name,\n",
                    "            \"split\": split,\n",
                    "            \"global_frame_id\": global_frame_id,\n",
                    "            \"timestamp\": timestamp\n",
                    "        })\n",
                    "        faiss_id_counter += 1\n",
                    "        \n",
                    "    return tasks\n",
                    "\n",
                    "tasks = build_keyframe_tasks(KEYFRAMES_DIR, METADATA_PATH, MAP_KEYFRAMES_DIR)\n",
                    "print(f\"Tổng số frame chuẩn bị chạy OCR: {len(tasks)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 5. Khởi tạo PaddleOCR Model (Tiếng Việt)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from paddleocr import PaddleOCR\n",
                    "logging.getLogger('ppocr').setLevel(logging.ERROR)\n",
                    "\n",
                    "print(\"Đang tải PaddleOCR (vi)...\")\n",
                    "try:\n",
                    "    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)\n",
                    "except Exception:\n",
                    "    ocr = PaddleOCR(use_textline_orientation=True, lang='vi')\n",
                    "print(\"PaddleOCR đã sẵn sàng!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 6. Chạy Trích xuất OCR & Tự động Lưu Kết Quả"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Nạp checkpoint nếu có\n",
                    "results_dict = {}\n",
                    "if os.path.exists(CHECKPOINT_JSON):\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            for item in json.load(f):\n",
                    "                results_dict[item[\"faiss_id\"]] = item\n",
                    "        print(f\"Đã nạp {len(results_dict)} kết quả từ checkpoint trước.\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Lỗi đọc checkpoint:\", e)\n",
                    "\n",
                    "SAVE_INTERVAL = 300\n",
                    "count = 0\n",
                    "\n",
                    "for item in tqdm(tasks, desc=\"Trích xuất OCR\"):\n",
                    "    fid = item[\"faiss_id\"]\n",
                    "    if fid in results_dict:\n",
                    "        continue\n",
                    "        \n",
                    "    img_path = item[\"image_path\"]\n",
                    "    try:\n",
                    "        res = ocr.ocr(img_path, cls=True)\n",
                    "        detected_texts = []\n",
                    "        \n",
                    "        if res and res[0]:\n",
                    "            for line in res[0]:\n",
                    "                if line and len(line) > 1 and line[1]:\n",
                    "                    text = str(line[1][0]).strip()\n",
                    "                    if text:\n",
                    "                        detected_texts.append(text)\n",
                    "                        \n",
                    "        if detected_texts:\n",
                    "            full_ocr_text = \" \".join(detected_texts)\n",
                    "            doc = {\n",
                    "                \"faiss_id\": fid,\n",
                    "                \"video_id\": item[\"video_id\"],\n",
                    "                \"frame_name\": item[\"frame_name\"],\n",
                    "                \"split\": item[\"split\"],\n",
                    "                \"global_frame_id\": item[\"global_frame_id\"],\n",
                    "                \"timestamp\": item[\"timestamp\"],\n",
                    "                \"ocr_text\": full_ocr_text,\n",
                    "                \"language\": \"vi\"\n",
                    "            }\n",
                    "            results_dict[fid] = doc\n",
                    "            \n",
                    "        count += 1\n",
                    "        if count >= SAVE_INTERVAL:\n",
                    "            os.makedirs(os.path.dirname(CHECKPOINT_JSON), exist_ok=True)\n",
                    "            with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "                json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "            count = 0\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"Lỗi xử lý ảnh {img_path}: {e}\")\n",
                    "\n",
                    "# Lưu file chính thức vào src/dict/ocr_results.json\n",
                    "final_results = list(results_dict.values())\n",
                    "os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(final_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n✅ HOÀN THÀNH! Đã lưu {len(final_results)} bản ghi OCR trực tiếp vào: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 7. Kiểm tra & Thống kê Kết quả"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if os.path.exists(OUTPUT_JSON):\n",
                    "    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:\n",
                    "        data = json.load(f)\n",
                    "    print(f\"Tổng số frame phát hiện text: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "    print(f\"\\n👉 File đã sẵn sàng tại: {OUTPUT_JSON}\")\n",
                    "    print(\"Bạn có thể chạy `python scripts/indexing/master_index_pipeline.py` để nạp vào Elasticsearch.\")"
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

def create_asr_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎙️ AIC 2026 - Video/Audio ASR Extraction Pipeline (Whisper / faster-whisper)\n",
                    "### 🚀 Tương thích: Google Colab GPU / VS Code Remote Colab / Local / Kaggle\n",
                    "\n",
                    "Notebook này tự động nạp cấu hình đường dẫn từ file **`.env`** của dự án, trích xuất giọng nói tiếng Việt bằng **faster-whisper (Large-v3-Turbo)**, tự động căn chỉnh với keyframe gần nhất và lưu trực tiếp file **`src/dict/asr_results.json`** chuẩn schema Elasticsearch `aic_asr` của Backend.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Cài đặt các thư viện cần thiết\n",
                    "!pip install faster-whisper transformers torch torchaudio python-dotenv tqdm pandas ffmpeg-python\n",
                    "!apt-get update && apt-get install -y ffmpeg"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Tự động Mount Google Drive nếu chạy trực tiếp trên Colab\n",
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "IN_COLAB = False\n",
                    "try:\n",
                    "    from google.colab import drive\n",
                    "    drive.mount('/content/drive')\n",
                    "    IN_COLAB = True\n",
                    "    print(\"✅ Đã kết nối Google Drive thành công!\")\n",
                    "except ImportError:\n",
                    "    print(\"ℹ️ Đang chạy trên môi trường Local / Remote SSH (không cần Colab Drive mount).\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 3. Tự động đọc cấu hình đường dẫn từ `.env`"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import glob\n",
                    "import json\n",
                    "import warnings\n",
                    "import pandas as pd\n",
                    "from tqdm import tqdm\n",
                    "import torch\n",
                    "from dotenv import load_dotenv\n",
                    "\n",
                    "warnings.filterwarnings(\"ignore\")\n",
                    "\n",
                    "# Tìm thư mục gốc dự án Backend\n",
                    "def find_project_root():\n",
                    "    curr = Path(os.getcwd()).resolve()\n",
                    "    for parent in [curr] + list(curr.parents):\n",
                    "        if (parent / \".env\").exists() or (parent / \"src\" / \"dict\").exists():\n",
                    "            return parent\n",
                    "    colab_drive_root = Path(\"/content/drive/MyDrive/AIC/Backend/Back-End\")\n",
                    "    if colab_drive_root.exists():\n",
                    "        return colab_drive_root\n",
                    "    return curr\n",
                    "\n",
                    "PROJECT_ROOT = find_project_root()\n",
                    "env_path = PROJECT_ROOT / \".env\"\n",
                    "\n",
                    "if env_path.exists():\n",
                    "    load_dotenv(dotenv_path=env_path)\n",
                    "    print(f\"✅ Đã load .env từ: {env_path}\")\n",
                    "else:\n",
                    "    print(f\"⚠️ Không tìm thấy file .env tại {env_path}, sử dụng cấu hình mặc định.\")\n",
                    "\n",
                    "# Lấy biến môi trường\n",
                    "ENV_METADATA_PATH = os.getenv(\"METADATA_PATH\", \"\")\n",
                    "\n",
                    "# Thiết lập các đường dẫn chuẩn theo backend\n",
                    "DICT_DIR = PROJECT_ROOT / \"src\" / \"dict\"\n",
                    "OUTPUT_JSON = DICT_DIR / \"asr_results.json\"\n",
                    "CHECKPOINT_JSON = DICT_DIR / \"asr_results_checkpoint.json\"\n",
                    "MAP_KEYFRAMES_DIR = DICT_DIR / \"map-keyframes\"\n",
                    "\n",
                    "# Xác định thư mục Videos / Media\n",
                    "MEDIA_DIR = None\n",
                    "candidate_media = [\n",
                    "    PROJECT_ROOT / \"src\" / \"data\" / \"Videos\",\n",
                    "    PROJECT_ROOT / \"src\" / \"data\" / \"videos\",\n",
                    "    Path(\"D:/AIC_Data/Videos\"),\n",
                    "    Path(\"D:/AIC_Data/videos\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC_Data/Videos\"),\n",
                    "    Path(\"/kaggle/input/your-videos-dataset/\")\n",
                    "]\n",
                    "for cand in candidate_media:\n",
                    "    if cand and cand.exists():\n",
                    "        MEDIA_DIR = cand\n",
                    "        break\n",
                    "\n",
                    "# Xác định file metadata\n",
                    "METADATA_PATH = None\n",
                    "candidate_meta = [\n",
                    "    Path(ENV_METADATA_PATH) if ENV_METADATA_PATH else None,\n",
                    "    DICT_DIR / \"metadata_clip.json\",\n",
                    "    Path(\"/content/drive/MyDrive/metadata_clip.json\"),\n",
                    "    Path(\"/kaggle/input/your-metadata-dataset/metadata_clip.json\")\n",
                    "]\n",
                    "for cand in candidate_meta:\n",
                    "    if cand and cand.exists():\n",
                    "        METADATA_PATH = cand\n",
                    "        break\n",
                    "\n",
                    "print(f\"\\n📁 CẤU HÌNH ĐƯỜNG DẪN HOẠT ĐỘNG:\")\n",
                    "print(f\"- Project Root:     {PROJECT_ROOT}\")\n",
                    "print(f\"- Media/Videos Dir: {MEDIA_DIR}\")\n",
                    "print(f\"- Metadata Path:    {METADATA_PATH}\")\n",
                    "print(f\"- Output Path:      {OUTPUT_JSON}\")\n",
                    "print(f\"- Map Keyframes:    {MAP_KEYFRAMES_DIR}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔗 4. Xây dựng Keyframe Timestamp Index để Alignment"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def build_keyframe_index(metadata_path=None, map_keyframes_dir=None):\n",
                    "    video_keyframe_map = {}\n",
                    "    \n",
                    "    if metadata_path and os.path.exists(metadata_path):\n",
                    "        print(f\"Đang nạp keyframes từ metadata: {metadata_path}\")\n",
                    "        with open(metadata_path, 'r', encoding='utf-8') as f:\n",
                    "            meta = json.load(f)\n",
                    "        for fid_str, info in meta.items():\n",
                    "            v_id = str(info.get(\"video_id\", \"\"))\n",
                    "            if not v_id:\n",
                    "                continue\n",
                    "            if v_id not in video_keyframe_map:\n",
                    "                video_keyframe_map[v_id] = []\n",
                    "            video_keyframe_map[v_id].append({\n",
                    "                \"faiss_id\": int(fid_str),\n",
                    "                \"timestamp\": float(info.get(\"timestamp\", 0.0)),\n",
                    "                \"frame_name\": str(info.get(\"frame_name\", \"\"))\n",
                    "            })\n",
                    "        for v_id in video_keyframe_map:\n",
                    "            video_keyframe_map[v_id].sort(key=lambda x: x[\"timestamp\"])\n",
                    "        print(f\"-> Đã nạp keyframes cho {len(video_keyframe_map)} videos.\")\n",
                    "        return video_keyframe_map\n",
                    "\n",
                    "    if map_keyframes_dir and os.path.exists(map_keyframes_dir):\n",
                    "        print(f\"Đang nạp keyframes từ thư mục CSV: {map_keyframes_dir}\")\n",
                    "        csv_files = glob.glob(f\"{map_keyframes_dir}/**/*.csv\", recursive=True)\n",
                    "        faiss_id_counter = 0\n",
                    "        for csv_f in csv_files:\n",
                    "            v_id = os.path.splitext(os.path.basename(csv_f))[0]\n",
                    "            video_keyframe_map[v_id] = []\n",
                    "            try:\n",
                    "                df = pd.read_csv(csv_f)\n",
                    "                for _, row in df.iterrows():\n",
                    "                    n_val = int(row['n'])\n",
                    "                    pts_time = float(row['pts_time'])\n",
                    "                    video_keyframe_map[v_id].append({\n",
                    "                        \"faiss_id\": faiss_id_counter,\n",
                    "                        \"timestamp\": pts_time,\n",
                    "                        \"frame_name\": f\"{n_val:04d}.webp\"\n",
                    "                    })\n",
                    "                    faiss_id_counter += 1\n",
                    "            except Exception:\n",
                    "                pass\n",
                    "        print(f\"-> Đã nạp keyframes cho {len(video_keyframe_map)} videos từ CSVs.\")\n",
                    "        return video_keyframe_map\n",
                    "        \n",
                    "    print(\"LƯU Ý: Không tìm thấy file metadata. Hệ thống sẽ căn chỉnh ước lượng.\")\n",
                    "    return video_keyframe_map\n",
                    "\n",
                    "def find_nearest_keyframe(target_time, video_id, keyframe_index):\n",
                    "    if video_id not in keyframe_index or not keyframe_index[video_id]:\n",
                    "        return {\"faiss_id\": 0, \"frame_name\": f\"frame_at_{int(target_time)}s.webp\"}\n",
                    "    frames = keyframe_index[video_id]\n",
                    "    return min(frames, key=lambda f: abs(f[\"timestamp\"] - target_time))\n",
                    "\n",
                    "keyframe_index = build_keyframe_index(METADATA_PATH, MAP_KEYFRAMES_DIR)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 5. Khởi tạo Whisper Model (Large-v3-Turbo)"
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
                    "print(f\"Đang tải faster-whisper [{MODEL_SIZE}] trên thiết bị [{DEVICE}] ({COMPUTE_TYPE})...\")\n",
                    "model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)\n",
                    "print(\"Whisper model đã sẵn sàng!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 6. Chạy Trích xuất ASR & Căn chỉnh Keyframe"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if not MEDIA_DIR or not os.path.exists(MEDIA_DIR):\n",
                    "    print(f\"❌ Thư mục media không tồn tại: {MEDIA_DIR}\")\n",
                    "    media_files = []\n",
                    "else:\n",
                    "    media_files = sorted(\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.mp4\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.mkv\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.mp3\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.wav\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.m4a\", recursive=True)\n",
                    "    )\n",
                    "print(f\"Tìm thấy {len(media_files)} media files để xử lý.\")\n",
                    "\n",
                    "# Nạp checkpoint đã xử lý\n",
                    "processed_videos = set()\n",
                    "asr_results = []\n",
                    "if os.path.exists(CHECKPOINT_JSON):\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            asr_results = json.load(f)\n",
                    "            processed_videos = {item[\"video_id\"] for item in asr_results}\n",
                    "        print(f\"Đã nạp {len(asr_results)} đoạn ASR từ checkpoint ({len(processed_videos)} videos).\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Lỗi nạp checkpoint:\", e)\n",
                    "\n",
                    "for media_path in tqdm(media_files, desc=\"Trích xuất ASR\"):\n",
                    "    video_id = os.path.splitext(os.path.basename(media_path))[0]\n",
                    "    if video_id in processed_videos:\n",
                    "        continue\n",
                    "        \n",
                    "    try:\n",
                    "        segments, info = model.transcribe(\n",
                    "            media_path,\n",
                    "            language=\"vi\",\n",
                    "            beam_size=5,\n",
                    "            word_timestamps=False,\n",
                    "            vad_filter=True,\n",
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
                    "        # Lưu checkpoint\n",
                    "        os.makedirs(os.path.dirname(CHECKPOINT_JSON), exist_ok=True)\n",
                    "        with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"Lỗi xử lý {media_path}: {e}\")\n",
                    "\n",
                    "# Lưu file chính thức vào src/dict/asr_results.json\n",
                    "os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n✅ HOÀN THÀNH! Đã lưu {len(asr_results)} đoạn hội thoại ASR trực tiếp vào: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 7. Kiểm tra & Thống kê Kết quả"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if os.path.exists(OUTPUT_JSON):\n",
                    "    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:\n",
                    "        data = json.load(f)\n",
                    "    print(f\"Tổng số câu ASR trích xuất: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "    print(f\"\\n👉 File đã sẵn sàng tại: {OUTPUT_JSON}\")\n",
                    "    print(\"Bạn có thể chạy `python scripts/indexing/master_index_pipeline.py` để nạp vào Elasticsearch.\")"
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
ocr_nb = create_ocr_notebook()
asr_nb = create_asr_notebook()

# Write to scripts/notebooks/kaggle_new/
os.makedirs(os.path.join(base_dir, "scripts", "notebooks", "kaggle_new"), exist_ok=True)
with open(os.path.join(base_dir, "scripts", "notebooks", "kaggle_new", "extract_ocr_kaggle.ipynb"), "w", encoding="utf-8") as f:
    json.dump(ocr_nb, f, indent=1, ensure_ascii=False)

with open(os.path.join(base_dir, "scripts", "notebooks", "kaggle_new", "extract_asr_kaggle.ipynb"), "w", encoding="utf-8") as f:
    json.dump(asr_nb, f, indent=1, ensure_ascii=False)

# Write to scripts/notebooks/
with open(os.path.join(base_dir, "scripts", "notebooks", "03_Extract_OCR_PaddleOCR.ipynb"), "w", encoding="utf-8") as f:
    json.dump(ocr_nb, f, indent=1, ensure_ascii=False)

with open(os.path.join(base_dir, "scripts", "notebooks", "04_Extract_ASR_Whisper.ipynb"), "w", encoding="utf-8") as f:
    json.dump(asr_nb, f, indent=1, ensure_ascii=False)

print("Updated all 4 notebook files with .env auto-detection and Colab Drive compatibility!")
