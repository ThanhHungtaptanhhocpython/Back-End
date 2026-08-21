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
                    "### 🎯 Kaggle GPU Edition - Lọc riêng các split từ **L21_a đến L30_a**\n",
                    "\n",
                    "Notebook này được tối ưu riêng cho bộ dataset **`HCMAI 2025 Extracted Keyframes`** trên Kaggle:\n",
                    "- Tự động quét và **chỉ lấy các thư mục từ `L21_a` đến `L30_a`** (bỏ qua `K01` -> `K20`).\n",
                    "- Nhận diện text tiếng Việt bằng **PaddleOCR** (GPU accelerated).\n",
                    "- Sử dụng module **`csv` chuẩn của Python** (không phụ thuộc pandas, không lo lỗi xung đột C-ABI/numpy).\n",
                    "- Tự động map `video_id`, `frame_name`, `pts_time/timestamp` chuẩn 100% schema Elasticsearch `aic_ocr` của Backend.\n",
                    "- Tự động lưu checkpoint liên tục phòng mất kết nối Kaggle.\n",
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
                    "# Cố định numpy < 2.0 để tương thích với paddlepaddle-gpu trên Kaggle\n",
                    "!pip install \"numpy<2.0.0\" paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 Pillow tqdm"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 2. Cấu hình & Tự động Phát hiện Thư mục Dataset `L21_a` -> `L30_a`"
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
                    "import csv\n",
                    "import re\n",
                    "import logging\n",
                    "from pathlib import Path\n",
                    "from tqdm import tqdm\n",
                    "from PIL import Image\n",
                    "\n",
                    "# Ẩn các log debug của PaddleOCR\n",
                    "logging.getLogger('ppocr').setLevel(logging.ERROR)\n",
                    "\n",
                    "# Danh sách các split cần lấy từ L21_a đến L30_a theo đúng cấu trúc dataset\n",
                    "TARGET_SPLITS = [\n",
                    "    \"L21_a\", \"L22_a\", \"L23_a\", \"L24_a\",\n",
                    "    \"L25_a\", \"L25_a1\", \"L25_b\",\n",
                    "    \"L26_a\", \"L26_b\", \"L26_c\", \"L26_d\", \"L26_e\",\n",
                    "    \"L27_a\", \"L28_a\", \"L29_a\", \"L30_a\"\n",
                    "]\n",
                    "\n",
                    "OUTPUT_JSON = Path(\"/kaggle/working/ocr_results.json\")\n",
                    "CHECKPOINT_JSON = Path(\"/kaggle/working/ocr_results_checkpoint.json\")\n",
                    "\n",
                    "# Tự động tìm thư mục gốc chứa các folder L21_a -> L30_a trong /kaggle/input/\n",
                    "DATASET_ROOT = None\n",
                    "search_base = \"/kaggle/input\"\n",
                    "\n",
                    "if os.path.exists(search_base):\n",
                    "    for root, dirs, _ in os.walk(search_base):\n",
                    "        # Kiểm tra xem thư mục này có chứa ít nhất 1 folder trong TARGET_SPLITS không\n",
                    "        found = [d for d in dirs if d in TARGET_SPLITS]\n",
                    "        if found:\n",
                    "            DATASET_ROOT = Path(root)\n",
                    "            break\n",
                    "\n",
                    "# Fallback nếu chạy local\n",
                    "if not DATASET_ROOT:\n",
                    "    if os.path.exists(\"D:/AIC_Data/Keyframes/keyframes_L21_onwards\"):\n",
                    "        DATASET_ROOT = Path(\"D:/AIC_Data/Keyframes/keyframes_L21_onwards\")\n",
                    "    else:\n",
                    "        DATASET_ROOT = Path(\"/kaggle/input/hcmai-2025-extracted-keyframes\")\n",
                    "\n",
                    "print(f\"📁 Thư mục Dataset gốc tìm thấy: {DATASET_ROOT}\")\n",
                    "print(f\"🎯 Số lượng splits mục tiêu: {len(TARGET_SPLITS)} ({', '.join(TARGET_SPLITS)})\")\n",
                    "print(f\"💾 File kết quả sẽ được lưu tại: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 3. Quét & Lọc chính xác các ảnh thuộc `L21_a` -> `L30_a`\n",
                    "Sử dụng module `csv` chuẩn của Python để tra cứu timestamp nhanh chóng và chính xác."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def load_map_keyframes_csv(csv_path):\n",
                    "    \"\"\"Đọc file map-keyframes CSV bằng module csv chuẩn của Python.\"\"\"\n",
                    "    mapping = {} # key: n (int) -> dict(pts_time, frame_idx)\n",
                    "    try:\n",
                    "        with open(csv_path, 'r', encoding='utf-8') as f:\n",
                    "            reader = csv.DictReader(f)\n",
                    "            for row in reader:\n",
                    "                try:\n",
                    "                    n_val = int(row.get('n', 0))\n",
                    "                    pts_time = float(row.get('pts_time', 0.0))\n",
                    "                    frame_idx = int(row.get('frame_idx', n_val))\n",
                    "                    mapping[n_val] = {\"pts_time\": pts_time, \"frame_idx\": frame_idx}\n",
                    "                except Exception:\n",
                    "                    pass\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "    return mapping\n",
                    "\n",
                    "def collect_target_keyframe_tasks(dataset_root, target_splits):\n",
                    "    tasks = []\n",
                    "    if not dataset_root.exists():\n",
                    "        print(f\"❌ Thư mục {dataset_root} không tồn tại!\")\n",
                    "        return tasks\n",
                    "\n",
                    "    # Tìm kiếm các map-keyframes CSVs nếu có trong input\n",
                    "    csv_cache = {}\n",
                    "    csv_files = glob.glob(\"/kaggle/input/**/map-keyframes*/**/*.csv\", recursive=True) + \\\n",
                    "                glob.glob(\"/kaggle/input/**/*.csv\", recursive=True)\n",
                    "    for cf in csv_files:\n",
                    "        v_name = os.path.splitext(os.path.basename(cf))[0]\n",
                    "        mapping = load_map_keyframes_csv(cf)\n",
                    "        if mapping:\n",
                    "            csv_cache[v_name] = mapping\n",
                    "            \n",
                    "    if csv_cache:\n",
                    "        print(f\"✅ Đã tìm thấy {len(csv_cache)} file CSV map-keyframes để đồng bộ timestamp!\")\n",
                    "\n",
                    "    faiss_id_counter = 0\n",
                    "    split_counts = {}\n",
                    "\n",
                    "    for split_name in target_splits:\n",
                    "        split_dir = dataset_root / split_name\n",
                    "        if not split_dir.exists():\n",
                    "            for d in os.listdir(dataset_root):\n",
                    "                if d.lower() == split_name.lower():\n",
                    "                    split_dir = dataset_root / d\n",
                    "                    break\n",
                    "                    \n",
                    "        if not split_dir.exists():\n",
                    "            print(f\"⚠️ Bỏ qua {split_name} (không tìm thấy thư mục)\")\n",
                    "            continue\n",
                    "\n",
                    "        image_paths = sorted(\n",
                    "            glob.glob(f\"{split_dir}/**/*.webp\", recursive=True) +\n",
                    "            glob.glob(f\"{split_dir}/**/*.jpg\", recursive=True) +\n",
                    "            glob.glob(f\"{split_dir}/**/*.png\", recursive=True)\n",
                    "        )\n",
                    "        \n",
                    "        split_counts[split_name] = len(image_paths)\n",
                    "        \n",
                    "        for img_path in image_paths:\n",
                    "            p = Path(img_path)\n",
                    "            frame_name = p.name\n",
                    "            \n",
                    "            # Trích xuất video_id (ví dụ: L21_V001 từ folder cha hoặc từ tên file)\n",
                    "            parent_name = p.parent.name\n",
                    "            match_vid = re.search(r\"(L\\d+_V\\d+)\", parent_name) or re.search(r\"(L\\d+_V\\d+)\", frame_name)\n",
                    "            if match_vid:\n",
                    "                video_id = match_vid.group(1)\n",
                    "            else:\n",
                    "                video_id = parent_name\n",
                    "                \n",
                    "            clean_split = split_name.split(\"_\")[0]  # L21, L22, L25...\n",
                    "            \n",
                    "            timestamp = 0.0\n",
                    "            global_frame_id = faiss_id_counter\n",
                    "            \n",
                    "            # Lấy timestamp chính xác từ CSV nếu có\n",
                    "            if video_id in csv_cache:\n",
                    "                v_map = csv_cache[video_id]\n",
                    "                num_match = re.search(r\"(\\d+)\", os.path.splitext(frame_name)[0])\n",
                    "                if num_match:\n",
                    "                    frame_num = int(num_match.group(1))\n",
                    "                    if frame_num in v_map:\n",
                    "                        timestamp = v_map[frame_num][\"pts_time\"]\n",
                    "                        global_frame_id = v_map[frame_num][\"frame_idx\"]\n",
                    "                    \n",
                    "            tasks.append({\n",
                    "                \"image_path\": img_path,\n",
                    "                \"faiss_id\": faiss_id_counter,\n",
                    "                \"video_id\": video_id,\n",
                    "                \"frame_name\": frame_name,\n",
                    "                \"split\": clean_split,\n",
                    "                \"split_folder\": split_name,\n",
                    "                \"global_frame_id\": global_frame_id,\n",
                    "                \"timestamp\": timestamp\n",
                    "            })\n",
                    "            faiss_id_counter += 1\n",
                    "\n",
                    "    print(\"\\n📊 THỐNG KÊ SỐ ẢNH MỖI SPLIT:\")\n",
                    "    for sp, cnt in split_counts.items():\n",
                    "        print(f\"  - {sp:8s}: {cnt:5d} frames\")\n",
                    "    print(f\"\\n🔥 TỔNG CỘNG: {len(tasks)} frames cần trích xuất OCR.\")\n",
                    "    return tasks\n",
                    "\n",
                    "tasks = collect_target_keyframe_tasks(DATASET_ROOT, TARGET_SPLITS)\n",
                    "if tasks:\n",
                    "    print(\"\\n👀 Mẫu dữ liệu đầu tiên:\", json.dumps(tasks[0], indent=2))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 4. Khởi tạo PaddleOCR Model (Tiếng Việt & Xoay chữ)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from paddleocr import PaddleOCR\n",
                    "print(\"Đang tải PaddleOCR (vi)...\")\n",
                    "try:\n",
                    "    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)\n",
                    "except Exception:\n",
                    "    ocr = PaddleOCR(use_textline_orientation=True, lang='vi')\n",
                    "print(\"✅ PaddleOCR đã sẵn sàng trên GPU!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. Thực hiện Trích xuất OCR & Tự động Lưu Checkpoint\n",
                    "- Tự động lưu checkpoint sau mỗi 300 ảnh.\n",
                    "- Nếu lỡ bị ngắt kết nối, chạy lại cell này sẽ tự động tiếp tục từ frame đã dừng."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Nạp checkpoint trước đó nếu có\n",
                    "results_dict = {}\n",
                    "if CHECKPOINT_JSON.exists():\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            for item in json.load(f):\n",
                    "                results_dict[item[\"faiss_id\"]] = item\n",
                    "        print(f\"🔄 Đã nạp {len(results_dict)} frame từ checkpoint cũ.\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Lỗi nạp checkpoint:\", e)\n",
                    "\n",
                    "SAVE_INTERVAL = 300\n",
                    "count = 0\n",
                    "\n",
                    "for item in tqdm(tasks, desc=\"Đang chạy OCR L21_a -> L30_a\"):\n",
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
                    "        print(f\"Lỗi xử lý frame {img_path}: {e}\")\n",
                    "\n",
                    "# Lưu file chính thức cuối cùng\n",
                    "final_results = list(results_dict.values())\n",
                    "os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(final_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n🎉 HOÀN TẤT TOÀN BỘ! Đã lưu {len(final_results)} bản ghi OCR vào: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 6. Kiểm tra & Hướng dẫn Tải về"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if OUTPUT_JSON.exists():\n",
                    "    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:\n",
                    "        data = json.load(f)\n",
                    "    print(f\"✅ Tổng số frame phát hiện có chữ OCR: {len(data)} / {len(tasks)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU KẾT QUẢ ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "        \n",
                    "    print(f\"\\n👉 File kết quả đã sẵn sàng tại: {OUTPUT_JSON}\")\n",
                    "    print(\"1. Ở cột bên phải Kaggle (tab Output), nhấn vào file `ocr_results.json` để tải về máy.\")\n",
                    "    print(\"2. Copy file vào thư mục Backend: `src/dict/ocr_results.json`.\")\n",
                    "    print(\"3. Chạy `python scripts/indexing/master_index_pipeline.py` để nạp vào Elasticsearch.\")"
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
                    "### 🎯 Kaggle GPU Edition - Lọc riêng các split từ **L21 đến L30**\n",
                    "\n",
                    "Notebook này trích xuất giọng nói tiếng Việt bằng **faster-whisper (Large-v3-Turbo)**:\n",
                    "- Sử dụng module **`csv` chuẩn của Python** (không phụ thuộc pandas, không lo lỗi xung đột C-ABI/numpy).\n",
                    "- Tự động căn chỉnh timestamp với keyframe gần nhất.\n",
                    "- Xuất file **`asr_results.json`** đúng 100% schema Elasticsearch `aic_asr` của Backend.\n",
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
                    "!pip install faster-whisper transformers torch torchaudio tqdm Pillow\n",
                    "!apt-get update && apt-get install -y ffmpeg"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚙️ 2. Cấu hình & Tự động Phát hiện Thư mục Videos/Media"
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
                    "import csv\n",
                    "import re\n",
                    "import warnings\n",
                    "from pathlib import Path\n",
                    "from tqdm import tqdm\n",
                    "import torch\n",
                    "\n",
                    "warnings.filterwarnings(\"ignore\")\n",
                    "\n",
                    "OUTPUT_JSON = Path(\"/kaggle/working/asr_results.json\")\n",
                    "CHECKPOINT_JSON = Path(\"/kaggle/working/asr_results_checkpoint.json\")\n",
                    "\n",
                    "# Tự động tìm thư mục Videos trong /kaggle/input/\n",
                    "MEDIA_DIR = None\n",
                    "possible_media = glob.glob(\"/kaggle/input/**/Videos*\", recursive=True) + glob.glob(\"/kaggle/input/**/videos*\", recursive=True)\n",
                    "if possible_media:\n",
                    "    MEDIA_DIR = Path(possible_media[0])\n",
                    "else:\n",
                    "    subdirs = [Path(\"/kaggle/input\") / d for d in os.listdir(\"/kaggle/input\")] if os.path.exists(\"/kaggle/input\") else []\n",
                    "    MEDIA_DIR = subdirs[0] if subdirs else Path(\"/kaggle/input/your-videos-dataset/\")\n",
                    "\n",
                    "print(f\"📁 Thư mục Videos tìm thấy: {MEDIA_DIR}\")\n",
                    "print(f\"💾 File kết quả sẽ được lưu tại: {OUTPUT_JSON}\")"
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
                    "def build_keyframe_index():\n",
                    "    video_keyframe_map = {}\n",
                    "    \n",
                    "    # Quét CSVs trong /kaggle/input/\n",
                    "    csv_files = glob.glob(\"/kaggle/input/**/map-keyframes*/**/*.csv\", recursive=True) + \\\n",
                    "                glob.glob(\"/kaggle/input/**/*.csv\", recursive=True)\n",
                    "    faiss_id_counter = 0\n",
                    "    \n",
                    "    for csv_f in csv_files:\n",
                    "        v_id = os.path.splitext(os.path.basename(csv_f))[0]\n",
                    "        try:\n",
                    "            with open(csv_f, 'r', encoding='utf-8') as f:\n",
                    "                reader = csv.DictReader(f)\n",
                    "                video_keyframe_map[v_id] = []\n",
                    "                for row in reader:\n",
                    "                    try:\n",
                    "                        n_val = int(row.get('n', 0))\n",
                    "                        pts_time = float(row.get('pts_time', 0.0))\n",
                    "                        video_keyframe_map[v_id].append({\n",
                    "                            \"faiss_id\": faiss_id_counter,\n",
                    "                            \"timestamp\": pts_time,\n",
                    "                            \"frame_name\": f\"{n_val:04d}.webp\"\n",
                    "                        })\n",
                    "                        faiss_id_counter += 1\n",
                    "                    except Exception:\n",
                    "                        pass\n",
                    "        except Exception:\n",
                    "            pass\n",
                    "            \n",
                    "    if video_keyframe_map:\n",
                    "        print(f\"✅ Đã nạp keyframes cho {len(video_keyframe_map)} videos từ CSVs.\")\n",
                    "    else:\n",
                    "        print(\"ℹ️ Không tìm thấy map-keyframes CSVs. Hệ thống sẽ căn chỉnh xấp xỉ.\")\n",
                    "        \n",
                    "    return video_keyframe_map\n",
                    "\n",
                    "def find_nearest_keyframe(target_time, video_id, keyframe_index):\n",
                    "    if video_id not in keyframe_index or not keyframe_index[video_id]:\n",
                    "        return {\"faiss_id\": 0, \"frame_name\": f\"frame_at_{int(target_time)}s.webp\"}\n",
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
                    "## 🤖 4. Khởi tạo Whisper Model (Large-v3-Turbo)"
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
                    "print(\"✅ Whisper model đã sẵn sàng!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. Thực hiện Trích xuất ASR & Căn chỉnh Keyframe (L21 -> L30)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if not MEDIA_DIR or not os.path.exists(MEDIA_DIR):\n",
                    "    print(f\"❌ Không tìm thấy thư mục media: {MEDIA_DIR}\")\n",
                    "    all_media = []\n",
                    "else:\n",
                    "    all_media = sorted(\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.mp4\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.mkv\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.mp3\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.wav\", recursive=True) +\n",
                    "        glob.glob(f\"{MEDIA_DIR}/**/*.m4a\", recursive=True)\n",
                    "    )\n",
                    "\n",
                    "# Lọc video từ L21 đến L30\n",
                    "media_files = []\n",
                    "for mf in all_media:\n",
                    "    basename = os.path.basename(mf)\n",
                    "    match = re.search(r\"L(2[1-9]|30)_V\\d+\", basename)\n",
                    "    if match:\n",
                    "        media_files.append(mf)\n",
                    "\n",
                    "if not media_files:\n",
                    "    media_files = all_media # fallback nếu tên không chứa tiền tố Lxx\n",
                    "    \n",
                    "print(f\"Tìm thấy {len(media_files)} video/audio thuộc dải L21 -> L30 để xử lý.\")\n",
                    "\n",
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
                    "        os.makedirs(os.path.dirname(CHECKPOINT_JSON), exist_ok=True)\n",
                    "        with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "            \n",
                    "    except Exception as e:\n",
                    "        print(f\"Lỗi xử lý {media_path}: {e}\")\n",
                    "\n",
                    "os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)\n",
                    "with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:\n",
                    "    json.dump(asr_results, f, ensure_ascii=False, indent=2)\n",
                    "\n",
                    "print(f\"\\n✅ HOÀN THÀNH! Đã lưu {len(asr_results)} đoạn hội thoại ASR vào: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 6. Kiểm tra & Hướng dẫn Tải về"
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
                    "    print(f\"\\n🎉 File kết quả đã sẵn sàng tại: {OUTPUT_JSON}\")\n",
                    "    print(\"👉 Bạn có thể tải file `asr_results.json` ở tab 'Output' bên phải màn hình Kaggle.\")\n",
                    "    print(\"👉 Sau khi tải về, copy vào thư mục `Backend/Back-End/src/dict/asr_results.json`.\")"
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

print("Successfully replaced pandas with standard csv module in all notebooks!")
