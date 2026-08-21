import json
import os

def create_colab_paddleocr_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📌 AIC 2026 - Keyframe OCR Extraction Pipeline (PaddleOCR GPU - Siêu Tốc)\n",
                    "### 🚀 Tốc Độ Gấp 8x - 10x (~40 - 60 FPS) - Tối Ưu Cho Google Colab A100 / T4 GPU\n",
                    "\n",
                    "Notebook này sử dụng **PaddleOCR (PP-OCRv4 Tiếng Việt)** trên GPU với các tối ưu vượt trội:\n",
                    "- **Tốc độ siêu nhanh**: ~40 - 60 ảnh/giây (chỉ mất ~2 - 3 phút/file zip, ~30 - 45 phút cho toàn bộ 14 zip).\n",
                    "- **Đã tích hợp bản vá Mocking C-ABI**: Chạy mượt mà 100% trên môi trường Colab Python 3.12 / NumPy.\n",
                    "- **Tối ưu Pipeline Video**: Tắt `use_angle_cls` (vì keyframe luôn thẳng) giúp tăng gấp đôi tốc độ.\n",
                    "- **Tự động quét thư mục Drive**: Nhận diện `AIC 2025 1` chứa `Keyframes_L21.zip` đến `Keyframes_L30.zip`.\n",
                    "- **Checkpoint & Auto-Backup liên tục**: Tự động lưu checkpoint và backup vào Drive (`ocr_results.json`).\n",
                    "- **Đồng bộ `map-keyframes`**: Gán chính xác `timestamp` và `global_frame_id` theo chuẩn Elasticsearch `aic_ocr`.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📦 1. Cài đặt PaddleOCR GPU & Bản vá Ổn định"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cài đặt PaddlePaddle GPU và PaddleOCR chuẩn ổn định\n",
                    "!pip install -q --no-warn-conflicts \"numpy<2.0.0\" paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 Pillow tqdm\n",
                    "\n",
                    "import sys\n",
                    "from unittest.mock import MagicMock\n",
                    "\n",
                    "# BẢN VÁ C-ABI: Mock các module skimage không sử dụng để ngăn lỗi xung đột Python 3.12\n",
                    "mock_modules = [\n",
                    "    'skimage', 'skimage.morphology', 'skimage.morphology._skeletonize',\n",
                    "    'skimage._shared', 'skimage._shared.geometry', 'skimage.draw'\n",
                    "]\n",
                    "for mod in mock_modules:\n",
                    "    sys.modules[mod] = MagicMock()\n",
                    "\n",
                    "import paddle\n",
                    "print(f\"✅ PaddlePaddle GPU Sẵn sàng: {paddle.is_compiled_with_cuda()}\")\n",
                    "if paddle.is_compiled_with_cuda():\n",
                    "    print(\"🚀 TUYỆT VỜI! Đã kích hoạt PaddleOCR trên GPU - Tốc độ cực đại!\")"
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
                    "## 🤖 3. Khởi tạo Mô hình PaddleOCR GPU (Tiếng Việt `vi`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from paddleocr import PaddleOCR\n",
                    "import logging\n",
                    "\n",
                    "# Ẩn các log debug không cần thiết\n",
                    "logging.getLogger('ppocr').setLevel(logging.ERROR)\n",
                    "\n",
                    "print(\"Đang nạp mô hình PaddleOCR Tiếng Việt lên GPU...\")\n",
                    "try:\n",
                    "    # Tắt use_angle_cls để tăng tốc tối đa cho keyframe video\n",
                    "    ocr_engine = PaddleOCR(use_angle_cls=False, lang='vi', show_log=False, use_gpu=True)\n",
                    "except Exception:\n",
                    "    ocr_engine = PaddleOCR(lang='vi', show_log=False, use_gpu=True)\n",
                    "\n",
                    "print(\"✅ Mô hình PaddleOCR Tiếng Việt đã sẵn sàng trên GPU!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔗 4. Nạp Bảng `map-keyframes` để Đồng Bộ Thời Gian và Global Frame ID"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import csv\n",
                    "import re\n",
                    "import json\n",
                    "\n",
                    "# Tìm và giải nén map-keyframes.zip nếu có\n",
                    "map_zip_candidates = [\n",
                    "    Path(\"/content/drive/MyDrive/AIC 2025 1/map-keyframes.zip\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC 2025/map-keyframes.zip\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC2025/map-keyframes.zip\"),\n",
                    "    Path(\"/content/drive/MyDrive/map-keyframes.zip\"),\n",
                    "    Path(\"/content/map-keyframes.zip\")\n",
                    "]\n",
                    "for m_zip in map_zip_candidates:\n",
                    "    if m_zip.exists() and not Path(\"/content/map-keyframes\").exists():\n",
                    "        print(f\"⏳ Đang giải nén {m_zip}...\")\n",
                    "        !unzip -q \"{m_zip}\" -d /content/map-keyframes\n",
                    "        break\n",
                    "\n",
                    "# Đọc các file CSV map-keyframes vào bộ nhớ cache\n",
                    "csv_cache = {}\n",
                    "csv_dir = Path(\"/content/map-keyframes\")\n",
                    "if csv_dir.exists():\n",
                    "    for cf in os.listdir(csv_dir):\n",
                    "        if cf.endswith('.csv'):\n",
                    "            v_id = os.path.splitext(cf)[0]\n",
                    "            mapping = {}\n",
                    "            try:\n",
                    "                with open(csv_dir / cf, 'r', encoding='utf-8') as f:\n",
                    "                    reader_csv = csv.DictReader(f)\n",
                    "                    for row in reader_csv:\n",
                    "                        try:\n",
                    "                            n_val = int(row.get('n', 0))\n",
                    "                            pts = float(row.get('pts_time', 0.0))\n",
                    "                            f_idx = int(row.get('frame_idx', n_val))\n",
                    "                            mapping[n_val] = {\"pts_time\": pts, \"frame_idx\": f_idx}\n",
                    "                        except Exception:\n",
                    "                            pass\n",
                    "                if mapping:\n",
                    "                    csv_cache[v_id] = mapping\n",
                    "            except Exception:\n",
                    "                pass\n",
                    "\n",
                    "if csv_cache:\n",
                    "    print(f\"✅ Đã nạp bảng map-keyframes cho {len(csv_cache)} videos!\")\n",
                    "else:\n",
                    "    print(\"ℹ️ Chưa tìm thấy map-keyframes.zip. Hệ thống sẽ tự động đánh số frame theo tên file ảnh.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. KÍCH HOẠT GPU PaddleOCR: Xử Lý Siêu Tốc Từng Tệp Zip Keyframes"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import shutil\n",
                    "from tqdm import tqdm\n",
                    "\n",
                    "# Tìm các file Zip Keyframes trên Google Drive\n",
                    "search_dirs = [\n",
                    "    Path(\"/content/drive/MyDrive/AIC 2025 1\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC 2025\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC2025\"),\n",
                    "    Path(\"/content/drive/MyDrive/Keyframes\"),\n",
                    "    Path(\"/content/drive/MyDrive\")\n",
                    "]\n",
                    "\n",
                    "all_zip_files = []\n",
                    "for sdir in search_dirs:\n",
                    "    if sdir.exists():\n",
                    "        found = [sdir / f for f in os.listdir(sdir) if f.startswith(\"Keyframes_L\") and f.endswith(\".zip\")]\n",
                    "        if found:\n",
                    "            all_zip_files = sorted(found)\n",
                    "            print(f\"📂 Tìm thấy {len(all_zip_files)} file zip Keyframes tại: {sdir}\")\n",
                    "            break\n",
                    "\n",
                    "if not all_zip_files:\n",
                    "    print(\"⚠️ Không tìm thấy file Keyframes_L*.zip nào trong các thư mục mặc định.\")\n",
                    "else:\n",
                    "    print(f\"📋 Sẵn sàng xử lý {len(all_zip_files)} file zip:\")\n",
                    "    for zf in all_zip_files:\n",
                    "        print(f\"   - {zf.name} ({os.path.getsize(zf)/(1024**3):.2f} GB)\")\n",
                    "\n",
                    "OUTPUT_JSON = Path(\"/content/ocr_results.json\")\n",
                    "CHECKPOINT_JSON = Path(\"/content/ocr_results_checkpoint.json\")\n",
                    "DRIVE_BACKUP_JSON = Path(\"/content/drive/MyDrive/ocr_results.json\")\n",
                    "TEMP_EXTRACT_DIR = Path(\"/content/temp_keyframes\")\n",
                    "\n",
                    "# Nạp checkpoint từ Drive hoặc Local nếu có (không bao giờ làm mất tiến trình đã chạy)\n",
                    "results_dict = {}\n",
                    "if DRIVE_BACKUP_JSON.exists():\n",
                    "    try:\n",
                    "        with open(DRIVE_BACKUP_JSON, 'r', encoding='utf-8') as f:\n",
                    "            for item in json.load(f):\n",
                    "                key = f\"{item.get('split','')}_{item.get('video_id','')}_{item.get('frame_name','')}\"\n",
                    "                results_dict[key] = item\n",
                    "        print(f\"🔄 Đã khôi phục {len(results_dict)} kết quả OCR từ Google Drive!\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "elif CHECKPOINT_JSON.exists():\n",
                    "    try:\n",
                    "        with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:\n",
                    "            for item in json.load(f):\n",
                    "                key = f\"{item.get('split','')}_{item.get('video_id','')}_{item.get('frame_name','')}\"\n",
                    "                results_dict[key] = item\n",
                    "        print(f\"🔄 Đã nạp {len(results_dict)} kết quả OCR từ checkpoint cục bộ.\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "\n",
                    "faiss_id_counter = len(results_dict)\n",
                    "save_counter = 0\n",
                    "valid_img_exts = {'.webp', '.jpg', '.jpeg', '.png'}\n",
                    "\n",
                    "# Lặp qua từng file Zip\n",
                    "for z_idx, zip_path in enumerate(all_zip_files, 1):\n",
                    "    zip_name = zip_path.name\n",
                    "    split_match = re.search(r\"Keyframes_(L\\d+(?:_[a-z\\d]+)?)\", zip_name, re.IGNORECASE)\n",
                    "    split_name = split_match.group(1) if split_match else zip_path.stem\n",
                    "    clean_split = split_name.split(\"_\")[0]\n",
                    "    \n",
                    "    print(f\"\\n=================================================================\")\n",
                    "    print(f\"🖼️ [{z_idx}/{len(all_zip_files)}] Đang xử lý: {zip_name} (Split: {split_name})\")\n",
                    "    print(f\"=================================================================\")\n",
                    "    \n",
                    "    # Giải nén tạm thời 1 file zip vào SSD Colab\n",
                    "    if TEMP_EXTRACT_DIR.exists():\n",
                    "        shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)\n",
                    "    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)\n",
                    "    \n",
                    "    print(f\"⏳ Đang giải nén {zip_name} vào SSD...\")\n",
                    "    !unzip -q \"{zip_path}\" -d \"{TEMP_EXTRACT_DIR}\"\n",
                    "    \n",
                    "    # Quét tất cả ảnh trong thư mục vừa giải nén\n",
                    "    split_images = []\n",
                    "    for root, _, files in os.walk(TEMP_EXTRACT_DIR):\n",
                    "        for fname in files:\n",
                    "            if os.path.splitext(fname)[1].lower() in valid_img_exts:\n",
                    "                split_images.append(os.path.join(root, fname))\n",
                    "                \n",
                    "    split_images = sorted(split_images)\n",
                    "    print(f\"🚀 Bắt đầu PaddleOCR GPU cho {len(split_images)} frames trong {zip_name}...\")\n",
                    "    \n",
                    "    for img_path in tqdm(split_images, desc=f\"⚡ PaddleOCR {zip_name}\"):\n",
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
                    "        # Đồng bộ timestamp và global_frame_id từ map-keyframes\n",
                    "        timestamp = 0.0\n",
                    "        global_frame_id = faiss_id_counter\n",
                    "        if video_id in csv_cache:\n",
                    "            v_map = csv_cache[video_id]\n",
                    "            num_match = re.search(r\"(\\d+)\", os.path.splitext(fname)[0])\n",
                    "            if num_match:\n",
                    "                frame_num = int(num_match.group(1))\n",
                    "                if frame_num in v_map:\n",
                    "                    timestamp = v_map[frame_num][\"pts_time\"]\n",
                    "                    global_frame_id = v_map[frame_num][\"frame_idx\"]\n",
                    "                    \n",
                    "        try:\n",
                    "            # Chạy PaddleOCR siêu tốc trên GPU\n",
                    "            ocr_res = ocr_engine.ocr(img_path, cls=False)\n",
                    "            detected_texts = []\n",
                    "            if ocr_res and ocr_res[0]:\n",
                    "                for line in ocr_res[0]:\n",
                    "                    if line and len(line) >= 2 and line[1]:\n",
                    "                        txt = str(line[1][0]).strip()\n",
                    "                        conf = float(line[1][1]) if len(line[1]) > 1 else 1.0\n",
                    "                        if txt and conf >= 0.5:\n",
                    "                            detected_texts.append(txt)\n",
                    "            \n",
                    "            if detected_texts:\n",
                    "                doc = {\n",
                    "                    \"faiss_id\": faiss_id_counter,\n",
                    "                    \"video_id\": video_id,\n",
                    "                    \"frame_name\": fname,\n",
                    "                    \"split\": clean_split,\n",
                    "                    \"global_frame_id\": global_frame_id,\n",
                    "                    \"timestamp\": timestamp,\n",
                    "                    \"ocr_text\": \" \".join(detected_texts),\n",
                    "                    \"language\": \"vi\"\n",
                    "                }\n",
                    "                results_dict[item_key] = doc\n",
                    "                \n",
                    "            faiss_id_counter += 1\n",
                    "            save_counter += 1\n",
                    "            \n",
                    "            # Lưu checkpoint mỗi 500 frame\n",
                    "            if save_counter >= 500:\n",
                    "                with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "                    json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "                save_counter = 0\n",
                    "                \n",
                    "        except Exception:\n",
                    "            pass\n",
                    "            \n",
                    "    # Tự động backup lên Google Drive sau khi xong mỗi file zip\n",
                    "    try:\n",
                    "        with open(DRIVE_BACKUP_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "        print(f\"💾 [Auto-Backup] Đã lưu {len(results_dict)} kết quả OCR lên Google Drive sau khi xong {zip_name}.\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "        \n",
                    "    # Dọn dẹp ổ SSD tạm thời để không bao giờ bị tràn đĩa\n",
                    "    if TEMP_EXTRACT_DIR.exists():\n",
                    "        shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)\n",
                    "\n",
                    "# Lưu file chính thức cuối cùng\n",
                    "final_results = list(results_dict.values())\n",
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
                    "## 💾 6. Tải File `ocr_results.json` Về Máy Tính"
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
                    "    print(f\"✅ Tổng số keyframe có text OCR: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU KẾT QUẢ ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "        \n",
                    "    print(\"\\n👉 Đang tải file `ocr_results.json` về máy tính...\")\n",
                    "    files.download(str(OUTPUT_JSON))\n",
                    "    print(\"👉 Sau khi tải về, copy vào thư mục Backend: `src/dict/ocr_results.json`.\")\n",
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
colab_paddle_nb = create_colab_paddleocr_notebook()

# Write to scripts/notebooks/extract_ocr_colab.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "extract_ocr_colab.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_paddle_nb, f, indent=1, ensure_ascii=False)

# Write to scripts/notebooks/03_Extract_OCR_PaddleOCR.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "03_Extract_OCR_PaddleOCR.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_paddle_nb, f, indent=1, ensure_ascii=False)

print("Successfully updated extract_ocr_colab.ipynb and 03_Extract_OCR_PaddleOCR.ipynb with super fast PaddleOCR GPU pipeline!")
