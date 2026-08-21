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
                    "### 🎯 Kaggle GPU Edition - Chạy Trực Tiếp Không Cần Chờ Quét (Streaming Mode)\n",
                    "\n",
                    "Notebook này được thiết kế theo cơ chế **Streaming trực tiếp**:\n",
                    "- **BỎ QUA hoàn toàn bước quét trước**: Bấm chạy là GPU hoạt động **NGAY LẬP TỨC từ giây đầu tiên**.\n",
                    "- Tự động duyệt qua từng split từ **`L21_a` đến `L30_a`**.\n",
                    "- Tự động lưu checkpoint liên tục mỗi 200 ảnh.\n",
                    "- Xuất file **`ocr_results.json`** đúng 100% schema Elasticsearch `aic_ocr` của Backend.\n",
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
                    "# 1. Cài đặt các thư viện (Đã fix force-reinstall tương thích cho Kaggle Python 3.12)\n",
                    "!pip install -q --upgrade --force-reinstall numpy==1.26.4 scipy==1.12.0 scikit-image==0.22.0\n",
                    "!pip install -q paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 Pillow tqdm"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚠️ Bước 1.1: Làm mới Kernel sau khi cài đặt thư viện\n",
                    "*(Nếu vừa chạy Cell 1 lần đầu, hãy chạy ô này để kernel nạp môi trường C-Runtime mới tinh)*"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import numpy as np\n",
                    "print(f\"Phiên bản NumPy hiện tại: {np.__version__}\")\n",
                    "if not np.__version__.startswith(\"1.26\"):\n",
                    "    print(\"🔄 Đang tự động restart kernel để áp dụng môi trường mới...\")\n",
                    "    os._exit(0)\n",
                    "else:\n",
                    "    print(\"✅ Môi trường NumPy 1.26.4 đã sẵn sàng 100%!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 2. Khởi tạo PaddleOCR Model trên GPU"
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
                    "import re\n",
                    "import logging\n",
                    "from pathlib import Path\n",
                    "from tqdm import tqdm\n",
                    "from PIL import Image\n",
                    "from paddleocr import PaddleOCR\n",
                    "\n",
                    "logging.getLogger('ppocr').setLevel(logging.ERROR)\n",
                    "print(\"Đang nạp mô hình PaddleOCR (vi) lên GPU...\")\n",
                    "try:\n",
                    "    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)\n",
                    "except Exception:\n",
                    "    ocr = PaddleOCR(use_textline_orientation=True, lang='vi')\n",
                    "print(\"✅ PaddleOCR đã nạp thành công lên GPU! Sẵn sàng trích xuất.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 3. KÍCH HOẠT GPU: Trực Tiếp Quét & OCR Từng Thư Mục `L21_a` -> `L30_a`\n",
                    "*(Chạy thẳng vào GPU ngay lập tức, không mất thời gian chờ chuẩn bị danh sách trước)*"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Danh sách các split cần lấy từ L21_a đến L30_a\n",
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
                    "for root, dirs, _ in os.walk(\"/kaggle/input\"):\n",
                    "    if any(d in TARGET_SPLITS for d in dirs):\n",
                    "        DATASET_ROOT = Path(root)\n",
                    "        break\n",
                    "\n",
                    "if not DATASET_ROOT:\n",
                    "    DATASET_ROOT = Path(\"/kaggle/input/hcmai-2025-extracted-keyframes\")\n",
                    "\n",
                    "print(f\"📁 Thư mục Dataset: {DATASET_ROOT}\")\n",
                    "\n",
                    "# Nạp checkpoint cũ nếu có\n",
                    "results_dict = {} # key: rel_path -> doc\n",
                    "if CHECKPOINT_JSON.exists():\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            for item in json.load(f):\n",
                    "                key = f\"{item.get('split','')}_{item.get('video_id','')}_{item.get('frame_name','')}\"\n",
                    "                results_dict[key] = item\n",
                    "        print(f\"🔄 Đã nạp {len(results_dict)} frame từ checkpoint cũ.\")\n",
                    "    except Exception as e:\n",
                    "        print(\"Lỗi checkpoint:\", e)\n",
                    "\n",
                    "valid_exts = {'.webp', '.jpg', '.jpeg', '.png'}\n",
                    "faiss_id_counter = len(results_dict)\n",
                    "save_counter = 0\n",
                    "total_processed = 0\n",
                    "total_detected = len(results_dict)\n",
                    "\n",
                    "# Duyệt và chạy OCR trực tiếp trên GPU cho từng split\n",
                    "for split_name in TARGET_SPLITS:\n",
                    "    split_dir = DATASET_ROOT / split_name\n",
                    "    if not split_dir.exists():\n",
                    "        for d in os.listdir(DATASET_ROOT):\n",
                    "            if d.lower() == split_name.lower():\n",
                    "                split_dir = DATASET_ROOT / d\n",
                    "                break\n",
                    "                \n",
                    "    if not split_dir.exists():\n",
                    "        continue\n",
                    "        \n",
                    "    clean_split = split_name.split(\"_\")[0] # L21, L22...\n",
                    "    print(f\"\\n🚀 [GPU] Đang xử lý thư mục: {split_name}...\")\n",
                    "    \n",
                    "    # Lấy danh sách ảnh trực tiếp của split này\n",
                    "    split_images = []\n",
                    "    for root, _, files in os.walk(split_dir):\n",
                    "        for fname in files:\n",
                    "            if os.path.splitext(fname)[1].lower() in valid_exts:\n",
                    "                split_images.append(os.path.join(root, fname))\n",
                    "                \n",
                    "    # Chạy OCR ngay lập tức với thanh tiến độ\n",
                    "    for img_path in tqdm(split_images, desc=f\"OCR {split_name}\"):\n",
                    "        p = Path(img_path)\n",
                    "        fname = p.name\n",
                    "        folder_name = p.parent.name\n",
                    "        \n",
                    "        m_vid = re.search(r\"(L\\d+_V\\d+)\", folder_name) or re.search(r\"(L\\d+_V\\d+)\", fname)\n",
                    "        video_id = m_vid.group(1) if m_vid else folder_name\n",
                    "        \n",
                    "        item_key = f\"{clean_split}_{video_id}_{fname}\"\n",
                    "        if item_key in results_dict:\n",
                    "            continue\n",
                    "            \n",
                    "        try:\n",
                    "            res = ocr.ocr(img_path, cls=True)\n",
                    "            detected_texts = []\n",
                    "            if res and res[0]:\n",
                    "                for line in res[0]:\n",
                    "                    if line and len(line) > 1 and line[1]:\n",
                    "                        text = str(line[1][0]).strip()\n",
                    "                        if text:\n",
                    "                            detected_texts.append(text)\n",
                    "                            \n",
                    "            if detected_texts:\n",
                    "                doc = {\n",
                    "                    \"faiss_id\": faiss_id_counter,\n",
                    "                    \"video_id\": video_id,\n",
                    "                    \"frame_name\": fname,\n",
                    "                    \"split\": clean_split,\n",
                    "                    \"global_frame_id\": faiss_id_counter,\n",
                    "                    \"timestamp\": 0.0,\n",
                    "                    \"ocr_text\": \" \".join(detected_texts),\n",
                    "                    \"language\": \"vi\"\n",
                    "                }\n",
                    "                results_dict[item_key] = doc\n",
                    "                total_detected += 1\n",
                    "                \n",
                    "            faiss_id_counter += 1\n",
                    "            total_processed += 1\n",
                    "            save_counter += 1\n",
                    "            \n",
                    "            # Tự động lưu checkpoint mỗi 200 ảnh\n",
                    "            if save_counter >= 200:\n",
                    "                os.makedirs(os.path.dirname(CHECKPOINT_JSON), exist_ok=True)\n",
                    "                with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "                    json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "                save_counter = 0\n",
                    "                \n",
                    "        except Exception as e:\n",
                    "            pass\n",
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
                    "## 📊 4. Kiểm tra & Hướng dẫn Tải về"
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
                    "    print(f\"✅ Tổng số frame phát hiện có chữ OCR: {len(data)}\")\n",
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

print("Updated OCR notebooks to Streaming Mode (zero-wait startup)!")
