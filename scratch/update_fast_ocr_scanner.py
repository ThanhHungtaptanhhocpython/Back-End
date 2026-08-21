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
                    "- Quét cực nhanh (dùng `os.walk` siêu tốc < 2 giây) các thư mục từ **`L21_a` đến `L30_a`**.\n",
                    "- Chạy GPU 100% bằng **PaddleOCR (GPU Accelerated)**.\n",
                    "- Lưu file kết quả **`ocr_results.json`** đúng 100% schema của Backend & Elasticsearch (`aic_ocr`).\n",
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
                    "## ⚙️ 2. Tự động Phát hiện Thư mục Dataset `L21_a` -> `L30_a`"
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
                    "# Tự động tìm thư mục gốc chứa các folder L21_a -> L30_a\n",
                    "DATASET_ROOT = None\n",
                    "search_base = \"/kaggle/input\"\n",
                    "\n",
                    "if os.path.exists(search_base):\n",
                    "    for root, dirs, _ in os.walk(search_base):\n",
                    "        found = [d for d in dirs if d in TARGET_SPLITS]\n",
                    "        if found:\n",
                    "            DATASET_ROOT = Path(root)\n",
                    "            break\n",
                    "\n",
                    "if not DATASET_ROOT:\n",
                    "    if os.path.exists(\"D:/AIC_Data/Keyframes/keyframes_L21_onwards\"):\n",
                    "        DATASET_ROOT = Path(\"D:/AIC_Data/Keyframes/keyframes_L21_onwards\")\n",
                    "    else:\n",
                    "        DATASET_ROOT = Path(\"/kaggle/input/hcmai-2025-extracted-keyframes\")\n",
                    "\n",
                    "print(f\"📁 Thư mục Dataset gốc: {DATASET_ROOT}\")\n",
                    "print(f\"🎯 Số lượng splits mục tiêu: {len(TARGET_SPLITS)}\")\n",
                    "print(f\"💾 File kết quả đầu ra: {OUTPUT_JSON}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 3. Quét & Lọc Siêu Tốc Các Ảnh `L21_a` -> `L30_a`\n",
                    "*(Dùng single-pass `os.walk` siêu nhanh chỉ mất 1-2 giây cho hàng chục ngàn ảnh)*"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def fast_collect_keyframe_tasks(dataset_root, target_splits):\n",
                    "    tasks = []\n",
                    "    if not dataset_root.exists():\n",
                    "        print(f\"❌ Thư mục {dataset_root} không tồn tại!\")\n",
                    "        return tasks\n",
                    "\n",
                    "    print(\"⏳ Đang quét danh sách file ảnh siêu tốc...\")\n",
                    "    faiss_id_counter = 0\n",
                    "    split_counts = {}\n",
                    "    valid_exts = {'.webp', '.jpg', '.jpeg', '.png'}\n",
                    "\n",
                    "    # Quét từng thư mục trong TARGET_SPLITS\n",
                    "    for split_name in target_splits:\n",
                    "        split_dir = dataset_root / split_name\n",
                    "        if not split_dir.exists():\n",
                    "            for d in os.listdir(dataset_root):\n",
                    "                if d.lower() == split_name.lower():\n",
                    "                    split_dir = dataset_root / d\n",
                    "                    break\n",
                    "                    \n",
                    "        if not split_dir.exists():\n",
                    "            continue\n",
                    "\n",
                    "        clean_split = split_name.split(\"_\")[0]  # L21, L22...\n",
                    "        count_this_split = 0\n",
                    "\n",
                    "        # Dùng os.walk 1 lần duy nhất - tốc độ cực nhanh\n",
                    "        for root, _, files in os.walk(split_dir):\n",
                    "            folder_name = os.path.basename(root)\n",
                    "            for fname in files:\n",
                    "                ext = os.path.splitext(fname)[1].lower()\n",
                    "                if ext not in valid_exts:\n",
                    "                    continue\n",
                    "                    \n",
                    "                img_path = os.path.join(root, fname)\n",
                    "                \n",
                    "                # Trích xuất video_id (ví dụ: L21_V001)\n",
                    "                match_vid = re.search(r\"(L\\d+_V\\d+)\", folder_name) or re.search(r\"(L\\d+_V\\d+)\", fname)\n",
                    "                video_id = match_vid.group(1) if match_vid else folder_name\n",
                    "                \n",
                    "                tasks.append({\n",
                    "                    \"image_path\": img_path,\n",
                    "                    \"faiss_id\": faiss_id_counter,\n",
                    "                    \"video_id\": video_id,\n",
                    "                    \"frame_name\": fname,\n",
                    "                    \"split\": clean_split,\n",
                    "                    \"split_folder\": split_name,\n",
                    "                    \"global_frame_id\": faiss_id_counter,\n",
                    "                    \"timestamp\": 0.0\n",
                    "                })\n",
                    "                faiss_id_counter += 1\n",
                    "                count_this_split += 1\n",
                    "                \n",
                    "        split_counts[split_name] = count_this_split\n",
                    "        print(f\"  ✅ Đã quét {split_name:8s}: {count_this_split:5d} frames\")\n",
                    "\n",
                    "    print(f\"\\n🔥 TỔNG CỘNG: {len(tasks)} frames sẵn sàng đưa vào GPU OCR!\")\n",
                    "    return tasks\n",
                    "\n",
                    "tasks = fast_collect_keyframe_tasks(DATASET_ROOT, TARGET_SPLITS)\n",
                    "if tasks:\n",
                    "    print(\"\\n👀 Mẫu dữ liệu đầu tiên:\", json.dumps(tasks[0], indent=2))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 4. Khởi tạo PaddleOCR Model trên GPU"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from paddleocr import PaddleOCR\n",
                    "print(\"Đang nạp mô hình PaddleOCR (vi) lên GPU...\")\n",
                    "try:\n",
                    "    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)\n",
                    "except Exception:\n",
                    "    ocr = PaddleOCR(use_textline_orientation=True, lang='vi')\n",
                    "print(\"✅ Mô hình PaddleOCR đã sẵn sàng trên GPU!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. KÍCH HOẠT GPU: Thực hiện Trích xuất OCR Toàn bộ Dataset\n",
                    "*(Ở bước này GPU sẽ hoạt động hết công suất và hiển thị thanh tiến độ `tqdm`)*"
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
                    "for item in tqdm(tasks, desc=\"🚀 GPU Đang chạy OCR L21_a -> L30_a\"):\n",
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

base_dir = r"c:\Users\Lenovo\Documents\GitHub\AIC\Backend\Back-End"
ocr_nb = create_ocr_notebook()

# Write to scripts/notebooks/kaggle_new/
os.makedirs(os.path.join(base_dir, "scripts", "notebooks", "kaggle_new"), exist_ok=True)
with open(os.path.join(base_dir, "scripts", "notebooks", "kaggle_new", "extract_ocr_kaggle.ipynb"), "w", encoding="utf-8") as f:
    json.dump(ocr_nb, f, indent=1, ensure_ascii=False)

# Write to scripts/notebooks/
with open(os.path.join(base_dir, "scripts", "notebooks", "03_Extract_OCR_PaddleOCR.ipynb"), "w", encoding="utf-8") as f:
    json.dump(ocr_nb, f, indent=1, ensure_ascii=False)

print("Updated OCR notebooks with lightning-fast os.walk scanner!")
