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
                    "\n",
                    "Notebook này thực hiện trích xuất chữ (OCR) từ các video keyframes bằng **PaddleOCR (Vietnamese)** và tạo ra file **`ocr_results.json`** với đúng schema chuẩn mà Backend và Elasticsearch (`aic_ocr`) yêu cầu.\n",
                    "\n",
                    "### 🎯 Schema đầu ra mong đợi cho Backend:\n",
                    "```json\n",
                    "[\n",
                    "  {\n",
                    "    \"faiss_id\": 10001,\n",
                    "    \"video_id\": \"L21_V001\",\n",
                    "    \"frame_name\": \"0001.webp\",\n",
                    "    \"split\": \"L21\",\n",
                    "    \"global_frame_id\": 0,\n",
                    "    \"timestamp\": 0.0,\n",
                    "    \"ocr_text\": \"BẢN TIN THỜI SỰ 19H\",\n",
                    "    \"language\": \"vi\"\n",
                    "  }\n",
                    "]\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Cài đặt các thư viện cần thiết\n",
                    "# Cố định numpy < 2.0 để tránh lỗi ABI với paddlepaddle\n",
                    "!pip install \"numpy<2.0.0\" paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 Pillow tqdm"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import glob\n",
                    "import json\n",
                    "import logging\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "from tqdm import tqdm\n",
                    "from PIL import Image\n",
                    "from paddleocr import PaddleOCR\n",
                    "\n",
                    "# Ẩn các log debug rác từ PaddleOCR\n",
                    "logging.getLogger('ppocr').setLevel(logging.ERROR)\n",
                    "\n",
                    "print(\"Tất cả thư viện đã được nạp thành công!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 2. Cấu hình Đường dẫn (Paths Configuration)\n",
                    "- `KEYFRAMES_DIR`: Thư mục chứa ảnh Keyframes trên Kaggle (hoặc máy local).\n",
                    "- `METADATA_PATH`: Đường dẫn tới `metadata_clip.json` (nếu có add vào Kaggle Input).\n",
                    "- `MAP_KEYFRAMES_DIR`: Thư mục chứa các file `map-keyframes/*.csv` (nếu có).\n",
                    "- `OUTPUT_JSON`: Đường dẫn file kết quả cuối cùng (`/kaggle/working/ocr_results.json`).\n",
                    "- `CHECKPOINT_JSON`: Đường dẫn file lưu tạm để phòng mất kết nối."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cấu hình đường dẫn - Hãy chỉnh lại cho phù hợp với Dataset của bạn trên Kaggle\n",
                    "KEYFRAMES_DIR = \"/kaggle/input/your-keyframes-dataset/\"  # Thư mục chứa ảnh .webp / .jpg\n",
                    "METADATA_PATH = \"/kaggle/input/your-metadata-dataset/metadata_clip.json\"  # metadata_clip.json nếu có\n",
                    "MAP_KEYFRAMES_DIR = \"/kaggle/input/your-map-keyframes-dataset/\" # Thư mục map-keyframes CSVs nếu có\n",
                    "\n",
                    "OUTPUT_JSON = \"/kaggle/working/ocr_results.json\"\n",
                    "CHECKPOINT_JSON = \"/kaggle/working/ocr_results_checkpoint.json\"\n",
                    "\n",
                    "BATCH_CHECKPOINT_INTERVAL = 500  # Tự động lưu checkpoint sau mỗi 500 ảnh\n",
                    "RESUME_FROM_CHECKPOINT = True    # Tiếp tục chạy tiếp từ checkpoint nếu bị ngắt quãng"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 3. Xây dựng Danh sách Keyframes & Metadata Index\n",
                    "Hàm dưới đây sẽ nạp metadata trực tiếp từ `metadata_clip.json` (nếu có), hoặc tự động quét cây thư mục keyframes và map với `map-keyframes/*.csv` để chuẩn hóa đầy đủ các trường:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def build_keyframe_tasks(keyframes_dir, metadata_path=None, map_keyframes_dir=None):\n",
                    "    \"\"\"Tạo danh sách task trích xuất OCR đầy đủ thông tin metadata theo đúng schema backend.\"\"\"\n",
                    "    tasks = []\n",
                    "    \n",
                    "    # Trường hợp 1: Có file metadata_clip.json chuẩn của dự án\n",
                    "    if metadata_path and os.path.exists(metadata_path):\n",
                    "        print(f\"Đang đọc metadata từ: {metadata_path}\")\n",
                    "        with open(metadata_path, 'r', encoding='utf-8') as f:\n",
                    "            meta = json.load(f)\n",
                    "        \n",
                    "        print(f\"Tổng số mục trong metadata_clip.json: {len(meta)}\")\n",
                    "        for fid_str, info in meta.items():\n",
                    "            faiss_id = int(fid_str)\n",
                    "            split = str(info.get(\"split\", \"\"))\n",
                    "            video_id = str(info.get(\"video_id\", \"\"))\n",
                    "            frame_name = str(info.get(\"frame_name\", \"\"))\n",
                    "            \n",
                    "            # Tìm đường dẫn ảnh thực tế trên disk\n",
                    "            candidates = [\n",
                    "                os.path.join(keyframes_dir, split, video_id, frame_name),\n",
                    "                os.path.join(keyframes_dir, video_id, frame_name),\n",
                    "                os.path.join(keyframes_dir, frame_name),\n",
                    "                os.path.join(keyframes_dir, f\"Keyframes_{split}\", video_id, frame_name),\n",
                    "            ]\n",
                    "            \n",
                    "            img_path = None\n",
                    "            for cand in candidates:\n",
                    "                if os.path.exists(cand):\n",
                    "                    img_path = cand\n",
                    "                    break\n",
                    "            \n",
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
                    "        print(f\"-> Khớp được {len(tasks)} ảnh tồn tại thực tế trên đĩa từ metadata.\")\n",
                    "        return tasks\n",
                    "\n",
                    "    # Trường hợp 2: Quét trực tiếp thư mục Keyframes nếu không có metadata_clip.json\n",
                    "    print(f\"Quét các file ảnh trực tiếp từ thư mục: {keyframes_dir}\")\n",
                    "    all_images = sorted(\n",
                    "        glob.glob(f\"{keyframes_dir}/**/*.webp\", recursive=True) +\n",
                    "        glob.glob(f\"{keyframes_dir}/**/*.jpg\", recursive=True) +\n",
                    "        glob.glob(f\"{keyframes_dir}/**/*.png\", recursive=True)\n",
                    "    )\n",
                    "    \n",
                    "    print(f\"Tìm thấy tổng cộng {len(all_images)} ảnh.\")\n",
                    "    \n",
                    "    # Cache map-keyframes CSVs nếu có\n",
                    "    csv_cache = {}\n",
                    "    if map_keyframes_dir and os.path.exists(map_keyframes_dir):\n",
                    "        for csv_file in glob.glob(f\"{map_keyframes_dir}/**/*.csv\", recursive=True):\n",
                    "            v_name = os.path.splitext(os.path.basename(csv_file))[0]\n",
                    "            try:\n",
                    "                df = pd.read_csv(csv_file)\n",
                    "                csv_cache[v_name] = df\n",
                    "            except Exception:\n",
                    "                pass\n",
                    "\n",
                    "    faiss_id_counter = 0\n",
                    "    for img_p in all_images:\n",
                    "        p = Path(img_p)\n",
                    "        frame_name = p.name\n",
                    "        video_id = p.parent.name\n",
                    "        split = p.parent.parent.name if len(p.parts) > 2 else \"\"\n",
                    "        \n",
                    "        # Chuẩn hóa split\n",
                    "        if \"Keyframes_\" in split:\n",
                    "            split = split.replace(\"Keyframes_\", \"\")\n",
                    "            \n",
                    "        timestamp = 0.0\n",
                    "        global_frame_id = faiss_id_counter\n",
                    "        \n",
                    "        # Thử lấy timestamp từ CSV\n",
                    "        if video_id in csv_cache:\n",
                    "            df = csv_cache[video_id]\n",
                    "            # Giả định tên frame dạng 0001.webp -> số thứ tự n = 1\n",
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
                    "if tasks:\n",
                    "    print(\"Mẫu task đầu tiên:\", json.dumps(tasks[0], indent=2))\n",
                    "else:\n",
                    "    print(\"CẢNH BÁO: Chưa tìm thấy ảnh nào. Vui lòng kiểm tra lại đường dẫn KEYFRAMES_DIR.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🚀 4. Khởi tạo PaddleOCR\n",
                    "Sử dụng mô hình Tiếng Việt (`lang='vi'`) với chế độ xoay dòng chữ tự động."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Đang khởi tạo PaddleOCR (vi)...\")\n",
                    "try:\n",
                    "    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)\n",
                    "except Exception:\n",
                    "    ocr = PaddleOCR(use_textline_orientation=True, lang='vi')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. Thực hiện Trích xuất OCR với Checkpointing & Auto-Save"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Nạp checkpoint nếu có\n",
                    "results_dict = {} # key là faiss_id để tránh trùng lặp\n",
                    "if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_JSON):\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            existing_data = json.load(f)\n",
                    "            for item in existing_data:\n",
                    "                results_dict[item[\"faiss_id\"]] = item\n",
                    "        print(f\"Đã nạp {len(results_dict)} kết quả từ checkpoint trước đó.\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Không thể đọc checkpoint:\", e)\n",
                    "\n",
                    "# Tiến hành xử lý\n",
                    "count_since_last_save = 0\n",
                    "\n",
                    "for item in tqdm(tasks, desc=\"Trích xuất OCR\"):\n",
                    "    fid = item[\"faiss_id\"]\n",
                    "    \n",
                    "    # Bỏ qua nếu đã xử lý\n",
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
                    "        count_since_last_save += 1\n",
                    "        \n",
                    "        # Lưu checkpoint định kỳ\n",
                    "        if count_since_last_save >= BATCH_CHECKPOINT_INTERVAL:\n",
                    "            with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "                json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "            count_since_last_save = 0\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"Lỗi xử lý ảnh {img_path}: {e}\")\n",
                    "\n",
                    "# Lưu file kết quả cuối cùng\n",
                    "final_results = list(results_dict.values())\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(final_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n✅ HOÀN TẤT! Đã lưu {len(final_results)} bản ghi OCR vào {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 6. Kiểm tra & Thống kê Kết quả"
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
                    "        \n",
                    "    print(f\"Tổng số frame có chứa chữ: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU KẾT QUẢ ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "        \n",
                    "    # Kiểm tra tính hợp lệ của schema\n",
                    "    required_fields = {\"faiss_id\", \"video_id\", \"frame_name\", \"split\", \"global_frame_id\", \"timestamp\", \"ocr_text\", \"language\"}\n",
                    "    if data and required_fields.issubset(data[0].keys()):\n",
                    "        print(\"\\n🎉 CHÚC MỪNG: Schema hoàn toàn khớp với Elasticsearch Backend `aic_ocr`!\")\n",
                    "    else:\n",
                    "        print(\"\\n⚠️ Cảnh báo: Schema còn thiếu một số trường.\")"
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
                    "\n",
                    "Notebook này thực hiện trích xuất giọng nói (ASR) từ video/audio bằng **Whisper / faster-whisper (Large-v3-Turbo)**, tự động căn chỉnh thời gian (timestamp alignment) với các keyframes gần nhất và xuất ra file **`asr_results.json`** theo đúng chuẩn Elasticsearch (`aic_asr`) của Backend.\n",
                    "\n",
                    "### 🎯 Schema đầu ra mong đợi cho Backend:\n",
                    "```json\n",
                    "[\n",
                    "  {\n",
                    "    \"video_id\": \"L21_V001\",\n",
                    "    \"start_time\": 0.0,\n",
                    "    \"end_time\": 2.5,\n",
                    "    \"text\": \"Chào mừng các bạn đến với bản tin thời sự...\",\n",
                    "    \"nearest_faiss_id\": 10001,\n",
                    "    \"nearest_frame_name\": \"0001.webp\"\n",
                    "  }\n",
                    "]\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Cài đặt các thư viện cần thiết\n",
                    "# Cài faster-whisper (nhanh hơn 4-5x so với vanilla whisper) và ffmpeg-python\n",
                    "!pip install faster-whisper transformers torch torchaudio tqdm pandas ffmpeg-python\n",
                    "!apt-get update && apt-get install -y ffmpeg"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import glob\n",
                    "import json\n",
                    "import subprocess\n",
                    "import warnings\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "from tqdm import tqdm\n",
                    "import torch\n",
                    "from faster_whisper import WhisperModel\n",
                    "\n",
                    "warnings.filterwarnings(\"ignore\")\n",
                    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"Device: {torch.cuda.get_device_name(0)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 2. Cấu hình Đường dẫn (Paths Configuration)\n",
                    "- `MEDIA_DIR`: Thư mục chứa video `.mp4` hoặc audio `.mp3`, `.wav`, `.m4a`.\n",
                    "- `METADATA_PATH`: Đường dẫn tới `metadata_clip.json` (nếu có).\n",
                    "- `MAP_KEYFRAMES_DIR`: Thư mục chứa các file CSV `map-keyframes/` (nếu có).\n",
                    "- `OUTPUT_JSON`: Đường dẫn file kết quả cuối cùng (`/kaggle/working/asr_results.json`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cấu hình đường dẫn trên Kaggle\n",
                    "MEDIA_DIR = \"/kaggle/input/your-videos-dataset/\"              # Thư mục chứa video .mp4 hoặc audio\n",
                    "METADATA_PATH = \"/kaggle/input/your-metadata-dataset/metadata_clip.json\" # File metadata_clip.json nếu có\n",
                    "MAP_KEYFRAMES_DIR = \"/kaggle/input/your-map-keyframes-dataset/\"          # Thư mục map-keyframes CSV nếu có\n",
                    "\n",
                    "OUTPUT_JSON = \"/kaggle/working/asr_results.json\"\n",
                    "CHECKPOINT_JSON = \"/kaggle/working/asr_results_checkpoint.json\"\n",
                    "\n",
                    "# Chọn model: 'large-v3-turbo' (khuyên dùng, chính xác và nhanh), 'large-v3', hoặc 'medium'\n",
                    "MODEL_SIZE = \"large-v3-turbo\"\n",
                    "COMPUTE_TYPE = \"float16\" if torch.cuda.is_available() else \"int8\"\n",
                    "RESUME_FROM_CHECKPOINT = True"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔗 3. Xây dựng Keyframe Timestamp Index để Alignment"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def build_keyframe_index(metadata_path=None, map_keyframes_dir=None):\n",
                    "    \"\"\"Tạo index tra cứu nhanh theo video_id -> list các frame [(timestamp, faiss_id, frame_name)].\"\"\"\n",
                    "    video_keyframe_map = {}\n",
                    "    \n",
                    "    # Cách 1: Nạp từ metadata_clip.json\n",
                    "    if metadata_path and os.path.exists(metadata_path):\n",
                    "        print(f\"Đang nạp keyframe index từ: {metadata_path}\")\n",
                    "        with open(metadata_path, 'r', encoding='utf-8') as f:\n",
                    "            meta = json.load(f)\n",
                    "            \n",
                    "        for fid_str, info in meta.items():\n",
                    "            v_id = str(info.get(\"video_id\", \"\"))\n",
                    "            if not v_id:\n",
                    "                continue\n",
                    "            if v_id not in video_keyframe_map:\n",
                    "                video_keyframe_map[v_id] = []\n",
                    "                \n",
                    "            video_keyframe_map[v_id].append({\n",
                    "                \"faiss_id\": int(fid_str),\n",
                    "                \"timestamp\": float(info.get(\"timestamp\", 0.0)),\n",
                    "                \"frame_name\": str(info.get(\"frame_name\", \"\"))\n",
                    "            })\n",
                    "            \n",
                    "        # Sắp xếp theo timestamp tăng dần cho mỗi video\n",
                    "        for v_id in video_keyframe_map:\n",
                    "            video_keyframe_map[v_id].sort(key=lambda x: x[\"timestamp\"])\n",
                    "            \n",
                    "        print(f\"-> Đã nạp keyframes cho {len(video_keyframe_map)} videos.\")\n",
                    "        return video_keyframe_map\n",
                    "\n",
                    "    # Cách 2: Nạp từ thư mục map-keyframes CSVs\n",
                    "    if map_keyframes_dir and os.path.exists(map_keyframes_dir):\n",
                    "        print(f\"Đang nạp keyframe index từ thư mục CSV: {map_keyframes_dir}\")\n",
                    "        csv_files = glob.glob(f\"{map_keyframes_dir}/**/*.csv\", recursive=True)\n",
                    "        faiss_id_counter = 0\n",
                    "        \n",
                    "        for csv_f in csv_files:\n",
                    "            v_id = os.path.splitext(os.path.basename(csv_f))[0]\n",
                    "            video_keyframe_map[v_id] = []\n",
                    "            try:\n",
                    "                df = pd.read_csv(csv_f)\n",
                    "                for _, row in df.iterrows():\n",
                    "                    n_val = int(row['n'])\n",
                    "                    pts_time = float(row['pts_time'])\n",
                    "                    frame_name = f\"{n_val:04d}.webp\" # hoặc .jpg\n",
                    "                    video_keyframe_map[v_id].append({\n",
                    "                        \"faiss_id\": faiss_id_counter,\n",
                    "                        \"timestamp\": pts_time,\n",
                    "                        \"frame_name\": frame_name\n",
                    "                    })\n",
                    "                    faiss_id_counter += 1\n",
                    "            except Exception as e:\n",
                    "                print(f\"Lỗi đọc CSV {csv_f}: {e}\")\n",
                    "                \n",
                    "        print(f\"-> Đã nạp keyframes cho {len(video_keyframe_map)} videos từ CSVs.\")\n",
                    "        return video_keyframe_map\n",
                    "        \n",
                    "    print(\"LƯU Ý: Không có metadata_clip.json hoặc map-keyframes. Hệ thống sẽ căn chỉnh xấp xỉ.\")\n",
                    "    return video_keyframe_map\n",
                    "\n",
                    "def find_nearest_keyframe(target_time, video_id, keyframe_index):\n",
                    "    \"\"\"Tìm keyframe có timestamp gần target_time nhất.\"\"\"\n",
                    "    if video_id not in keyframe_index or not keyframe_index[video_id]:\n",
                    "        # Trả về fallback document nếu không tìm thấy keyframe mapping\n",
                    "        return {\"faiss_id\": 0, \"frame_name\": f\"frame_at_{int(target_time)}s.webp\"}\n",
                    "        \n",
                    "    frames = keyframe_index[video_id]\n",
                    "    # Tìm kiếm nhị phân hoặc lặp tìm min diff\n",
                    "    best_frame = min(frames, key=lambda f: abs(f[\"timestamp\"] - target_time))\n",
                    "    return best_frame\n",
                    "\n",
                    "keyframe_index = build_keyframe_index(METADATA_PATH, MAP_KEYFRAMES_DIR)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 4. Khởi tạo Mô hình Whisper"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Đang tải Whisper Model [{MODEL_SIZE}] với Compute Type [{COMPUTE_TYPE}]...\")\n",
                    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                    "model = WhisperModel(MODEL_SIZE, device=device, compute_type=COMPUTE_TYPE)\n",
                    "print(\"Whisper model đã sẵn sàng!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. Thực hiện Trích xuất ASR & Căn chỉnh Keyframe"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Tìm tất cả media files\n",
                    "media_files = sorted(\n",
                    "    glob.glob(f\"{MEDIA_DIR}/**/*.mp4\", recursive=True) +\n",
                    "    glob.glob(f\"{MEDIA_DIR}/**/*.mkv\", recursive=True) +\n",
                    "    glob.glob(f\"{MEDIA_DIR}/**/*.mp3\", recursive=True) +\n",
                    "    glob.glob(f\"{MEDIA_DIR}/**/*.wav\", recursive=True) +\n",
                    "    glob.glob(f\"{MEDIA_DIR}/**/*.m4a\", recursive=True)\n",
                    ")\n",
                    "\n",
                    "print(f\"Tìm thấy {len(media_files)} media files để trích xuất ASR.\")\n",
                    "\n",
                    "# Nạp checkpoint đã xử lý nếu có\n",
                    "processed_videos = set()\n",
                    "asr_results = []\n",
                    "\n",
                    "if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_JSON):\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            asr_results = json.load(f)\n",
                    "            processed_videos = {item[\"video_id\"] for item in asr_results}\n",
                    "        print(f\"Đã nạp {len(asr_results)} đoạn ASR của {len(processed_videos)} videos từ checkpoint.\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Không thể nạp checkpoint:\", e)\n",
                    "\n",
                    "# Loop xử lý từng media file\n",
                    "for media_path in tqdm(media_files, desc=\"Trích xuất ASR\"):\n",
                    "    video_id = os.path.splitext(os.path.basename(media_path))[0]\n",
                    "    \n",
                    "    if video_id in processed_videos:\n",
                    "        continue\n",
                    "        \n",
                    "    try:\n",
                    "        # faster-whisper có thể nhận trực tiếp file video / audio mà không cần trích xuất wav thủ công\n",
                    "        segments, info = model.transcribe(\n",
                    "            media_path,\n",
                    "            language=\"vi\",\n",
                    "            beam_size=5,\n",
                    "            word_timestamps=False,\n",
                    "            vad_filter=True, # Bỏ qua các đoạn im lặng\n",
                    "            vad_parameters=dict(min_silence_duration_ms=500)\n",
                    "        )\n",
                    "        \n",
                    "        video_segments = []\n",
                    "        for seg in segments:\n",
                    "            start_time = round(float(seg.start), 3)\n",
                    "            end_time = round(float(seg.end), 3)\n",
                    "            text = seg.text.strip()\n",
                    "            \n",
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
                    "        # Lưu checkpoint sau mỗi video\n",
                    "        with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"Lỗi xử lý file {media_path}: {e}\")\n",
                    "\n",
                    "# Lưu file kết quả cuối cùng\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n✅ HOÀN TẤT! Đã trích xuất {len(asr_results)} đoạn hội thoại và lưu vào {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 6. Kiểm tra & Thống kê Kết quả"
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
                    "        \n",
                    "    print(f\"Tổng số câu ASR: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU KẾT QUẢ ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "        \n",
                    "    # Kiểm tra tính hợp lệ của schema\n",
                    "    required_fields = {\"video_id\", \"start_time\", \"end_time\", \"text\", \"nearest_faiss_id\", \"nearest_frame_name\"}\n",
                    "    if data and required_fields.issubset(data[0].keys()):\n",
                    "        print(\"\\n🎉 CHÚC MỪNG: Schema hoàn toàn khớp với Elasticsearch Backend `aic_asr`!\")\n",
                    "    else:\n",
                    "        print(\"\\n⚠️ Cảnh báo: Schema còn thiếu một số trường.\")"
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

# Generate files
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

print("Created all 4 notebook files successfully!")
