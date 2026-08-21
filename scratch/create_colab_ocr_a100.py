import json
import os

def create_colab_ocr_a100_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📌 AIC 2026 - Keyframe OCR Extraction Pipeline (EasyOCR - PyTorch GPU)\n",
                    "### 🚀 Google Colab A100 / T4 GPU Edition - Chạy Trực Tiếp Không Cần Chờ Quét\n",
                    "\n",
                    "Notebook này trích xuất văn bản tiếng Việt từ Keyframes bằng **EasyOCR (PyTorch GPU)**:\n",
                    "- **Chạy siêu tốc trên GPU A100 / T4**: Dùng PyTorch Native, không bị lỗi C-ABI hay xung đột thư viện.\n",
                    "- **Cơ chế Streaming Trực tiếp**: Bỏ qua hoàn toàn bước quét trước, bấm chạy là GPU hoạt động **NGAY TỪ GIÂY ĐẦU TIÊN**.\n",
                    "- **Hỗ trợ cả Thư mục & File Zip từ Google Drive**: Tự động giải nén tuần tự từng phần để không bao giờ bị đầy ổ cứng Colab.\n",
                    "- **Đồng bộ với `map-keyframes`**: Gán chính xác `timestamp` và `global_frame_id` theo chuẩn BTC.\n",
                    "- **Tự động lưu vào Google Drive**: Lưu file kết quả `ocr_results.json` thẳng vào Drive của bạn.\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📦 1. Cài đặt các thư viện cần thiết (Chỉ mất 5 giây)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q easyocr tqdm Pillow\n",
                    "\n",
                    "import torch\n",
                    "print(f\"✅ PyTorch CUDA Sẵn sàng: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    gpu_name = torch.cuda.get_device_name(0)\n",
                    "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                    "    print(f\"⚡ GPU Thiết bị: {gpu_name} ({vram_gb:.1f} GB VRAM)\")\n",
                    "    if \"A100\" in gpu_name.upper():\n",
                    "        print(\"🚀 TUYỆT VỜI! Đã kích hoạt NVIDIA A100 - Sẵn sàng chạy OCR siêu tốc!\")"
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
                    "## 🤖 3. Khởi tạo Mô hình EasyOCR trên GPU (Tiếng Việt `vi` + Tiếng Anh `en`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import easyocr\n",
                    "\n",
                    "print(\"Đang nạp mô hình EasyOCR (Tiếng Việt & Tiếng Anh) lên GPU...\")\n",
                    "reader = easyocr.Reader(['vi', 'en'], gpu=True, verbose=False)\n",
                    "print(\"✅ EasyOCR đã sẵn sàng hoạt động trên GPU!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔗 4. Nạp Bảng `map-keyframes` để Đồng Bộ Thời Gian (Nếu có)"
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
                    "    Path(\"/content/drive/MyDrive/map-keyframes.zip\"),\n",
                    "    Path(\"/content/drive/MyDrive/AIC2025/map-keyframes.zip\"),\n",
                    "    Path(\"/content/map-keyframes.zip\")\n",
                    "]\n",
                    "for m_zip in map_zip_candidates:\n",
                    "    if m_zip.exists() and not Path(\"/content/map-keyframes\").exists():\n",
                    "        print(f\"⏳ Đang giải nén {m_zip.name}...\")\n",
                    "        !unzip -q \"{m_zip}\" -d /content/map-keyframes\n",
                    "        break\n",
                    "\n",
                    "# Đọc các file CSV map-keyframes vào bộ nhớ\n",
                    "csv_cache = {} # key: video_id -> dict(frame_num -> dict(pts_time, frame_idx))\n",
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
                    "    print(\"ℹ️ Chưa có map-keyframes CSVs. Hệ thống sẽ đánh số frame tự động theo tên file.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚡ 5. KÍCH HOẠT A100 GPU: Trực Tiếp Trích Xuất OCR Từng Thư Mục `L21_a` -> `L30_a`\n",
                    "*(Chạy thẳng vào GPU ngay lập tức, không mất thời gian chờ quét danh sách)*"
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
                    "TARGET_SPLITS = [\n",
                    "    \"L21_a\", \"L22_a\", \"L23_a\", \"L24_a\",\n",
                    "    \"L25_a\", \"L25_a1\", \"L25_b\",\n",
                    "    \"L26_a\", \"L26_b\", \"L26_c\", \"L26_d\", \"L26_e\",\n",
                    "    \"L27_a\", \"L28_a\", \"L29_a\", \"L30_a\"\n",
                    "]\n",
                    "\n",
                    "OUTPUT_JSON = Path(\"/content/ocr_results.json\")\n",
                    "CHECKPOINT_JSON = Path(\"/content/ocr_results_checkpoint.json\")\n",
                    "DRIVE_BACKUP_JSON = Path(\"/content/drive/MyDrive/ocr_results.json\")\n",
                    "TEMP_EXTRACT_DIR = Path(\"/content/temp_keyframes\")\n",
                    "\n",
                    "# Nạp checkpoint từ Drive hoặc Local nếu có\n",
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
                    "# Duyệt qua từng split theo thứ tự\n",
                    "for split_idx, split_name in enumerate(TARGET_SPLITS, 1):\n",
                    "    clean_split = split_name.split(\"_\")[0] # L21, L22...\n",
                    "    print(f\"\\n=================================================================\")\n",
                    "    print(f\"🖼️ [{split_idx}/{len(TARGET_SPLITS)}] Đang xử lý thư mục: {split_name}\")\n",
                    "    print(f\"=================================================================\")\n",
                    "    \n",
                    "    # Kiểm tra xem có thư mục giải nén sẵn hay file zip trên Drive\n",
                    "    split_dir = None\n",
                    "    candidate_dirs = [\n",
                    "        Path(f\"/content/keyframes/{split_name}\"),\n",
                    "        Path(f\"/content/drive/MyDrive/{split_name}\"),\n",
                    "        Path(f\"/content/drive/MyDrive/AIC2025/{split_name}\"),\n",
                    "        Path(f\"/content/drive/MyDrive/Keyframes/{split_name}\"),\n",
                    "        Path(f\"/content/drive/MyDrive/Keyframes_L21_onwards/{split_name}\")\n",
                    "    ]\n",
                    "    for cd in candidate_dirs:\n",
                    "        if cd.exists():\n",
                    "            split_dir = cd\n",
                    "            break\n",
                    "            \n",
                    "    # Nếu là file zip (ví dụ: Keyframes_L21_a.zip hoặc L21_a.zip)\n",
                    "    if not split_dir:\n",
                    "        candidate_zips = [\n",
                    "            Path(f\"/content/drive/MyDrive/Keyframes_{split_name}.zip\"),\n",
                    "            Path(f\"/content/drive/MyDrive/{split_name}.zip\"),\n",
                    "            Path(f\"/content/drive/MyDrive/AIC2025/{split_name}.zip\")\n",
                    "        ]\n",
                    "        for cz in candidate_zips:\n",
                    "            if cz.exists():\n",
                    "                print(f\"⏳ Đang giải nén tạm {cz.name} vào SSD Colab...\")\n",
                    "                if TEMP_EXTRACT_DIR.exists():\n",
                    "                    shutil.rmtree(TEMP_EXTRACT_DIR)\n",
                    "                TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)\n",
                    "                !unzip -q \"{cz}\" -d \"{TEMP_EXTRACT_DIR}\"\n",
                    "                split_dir = TEMP_EXTRACT_DIR\n",
                    "                break\n",
                    "                \n",
                    "    if not split_dir or not split_dir.exists():\n",
                    "        print(f\"⚠️ Bỏ qua {split_name} (chưa tìm thấy thư mục hoặc file zip trên Drive)\")\n",
                    "        continue\n",
                    "        \n",
                    "    # Quét ảnh của split này\n",
                    "    split_images = []\n",
                    "    for root, _, files in os.walk(split_dir):\n",
                    "        for fname in files:\n",
                    "            if os.path.splitext(fname)[1].lower() in valid_img_exts:\n",
                    "                split_images.append(os.path.join(root, fname))\n",
                    "                \n",
                    "    split_images = sorted(split_images)\n",
                    "    print(f\"🚀 GPU bắt đầu trích xuất {len(split_images)} frames trong {split_name}...\")\n",
                    "    \n",
                    "    for img_path in tqdm(split_images, desc=f\"EasyOCR {split_name}\"):\n",
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
                    "            # Chạy EasyOCR trên GPU\n",
                    "            res = reader.readtext(img_path, detail=0)\n",
                    "            detected_texts = [str(t).strip() for t in res if str(t).strip()]\n",
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
                    "            # Lưu checkpoint mỗi 200 frame\n",
                    "            if save_counter >= 200:\n",
                    "                with open(CHECKPOINT_JSON, 'w', encoding='utf-8') as f:\n",
                    "                    json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "                save_counter = 0\n",
                    "                \n",
                    "        except Exception:\n",
                    "            pass\n",
                    "            \n",
                    "    # Tự động backup lên Drive sau mỗi split\n",
                    "    try:\n",
                    "        with open(DRIVE_BACKUP_JSON, 'w', encoding='utf-8') as f:\n",
                    "            json.dump(list(results_dict.values()), f, ensure_ascii=False, indent=2)\n",
                    "        print(f\"💾 [Auto-Backup] Đã lưu {len(results_dict)} kết quả OCR lên Google Drive sau khi xong {split_name}.\")\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "        \n",
                    "    # Dọn dẹp ổ SSD nếu dùng file zip tạm\n",
                    "    if split_dir == TEMP_EXTRACT_DIR and TEMP_EXTRACT_DIR.exists():\n",
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
                    "## 💾 6. Lưu Trực Tiếp File `ocr_results.json` Vào Google Drive"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import shutil\n",
                    "\n",
                    "DRIVE_FINAL_PATH = Path(\"/content/drive/MyDrive/ocr_results.json\")\n",
                    "\n",
                    "if OUTPUT_JSON.exists():\n",
                    "    shutil.copy(str(OUTPUT_JSON), str(DRIVE_FINAL_PATH))\n",
                    "    sz_mb = os.path.getsize(DRIVE_FINAL_PATH) / (1024 * 1024)\n",
                    "    print(f\"✅ ĐÃ LƯU THÀNH CÔNG VÀO GOOGLE DRIVE: {DRIVE_FINAL_PATH}\")\n",
                    "    print(f\"📊 Kích thước file: {sz_mb:.2f} MB\")\n",
                    "    print(\"👉 Bạn có thể vào Google Drive và tải file `ocr_results.json` về bất kỳ lúc nào!\")\n",
                    "else:\n",
                    "    print(\"❌ Chưa tìm thấy file ocr_results.json để lưu.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📥 7. Tùy Chọn: Tải File Trực Tiếp Về Trình Duyệt Máy Tính"
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
                    "    print(f\"✅ Tổng số frame phát hiện có chữ OCR: {len(data)}\")\n",
                    "    if data:\n",
                    "        print(\"\\n--- 3 MẪU KẾT QUẢ ĐẦU TIÊN ---\")\n",
                    "        print(json.dumps(data[:3], indent=2, ensure_ascii=False))\n",
                    "        \n",
                    "    print(\"\\n👉 Đang tải file `ocr_results.json` về máy tính qua trình duyệt...\")\n",
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
colab_ocr_nb = create_colab_ocr_a100_notebook()

# Write to scripts/notebooks/extract_ocr_colab.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "extract_ocr_colab.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_ocr_nb, f, indent=1, ensure_ascii=False)

# Write to scripts/notebooks/03_Extract_OCR_PaddleOCR.ipynb
with open(os.path.join(base_dir, "scripts", "notebooks", "03_Extract_OCR_PaddleOCR.ipynb"), "w", encoding="utf-8") as f:
    json.dump(colab_ocr_nb, f, indent=1, ensure_ascii=False)

print("Created extract_ocr_colab.ipynb specifically optimized for Google Colab GPU A100 with zero-wait streaming!")
