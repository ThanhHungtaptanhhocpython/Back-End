import json
import shutil
import os

base_dir = r"c:\Users\Lenovo\Documents\GitHub\AIC\Backend\Back-End"
scripts_dir = os.path.join(base_dir, "scripts", "notebooks")

ocr_colab = os.path.join(scripts_dir, "extract_ocr_colab.ipynb")
ocr_legacy = os.path.join(scripts_dir, "03_Extract_OCR_PaddleOCR.ipynb")

asr_colab = os.path.join(scripts_dir, "extract_asr_colab.ipynb")
asr_legacy = os.path.join(scripts_dir, "04_Extract_ASR_Whisper.ipynb")

# Sync OCR
shutil.copy(ocr_colab, ocr_legacy)
print("Synced extract_ocr_colab.ipynb -> 03_Extract_OCR_PaddleOCR.ipynb")

# Update ASR search_dirs to include AIC 2025 1
with open(asr_colab, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "search_dirs = [" in src and "AIC 2025 1" not in src:
            old_s = 'search_dirs = [\n    Path("/content/drive/MyDrive/video_batch_1"),\n    Path("/content/drive/MyDrive/AIC2025/video_batch_1"),\n    Path("/content/drive/MyDrive/AIC_Data/video_batch_1"),\n    Path("/content/drive/MyDrive")\n]'
            new_s = 'search_dirs = [\n    Path("/content/drive/MyDrive/AIC 2025 1/video_batch_1"),\n    Path("/content/drive/MyDrive/AIC 2025 1"),\n    Path("/content/drive/MyDrive/video_batch_1"),\n    Path("/content/drive/MyDrive/AIC2025/video_batch_1"),\n    Path("/content/drive/MyDrive/AIC_Data/video_batch_1"),\n    Path("/content/drive/MyDrive")\n]'
            src = src.replace(old_s, new_s)
            cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + ([src.split("\n")[-1]] if src.split("\n")[-1] else [])

with open(asr_colab, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

shutil.copy(asr_colab, asr_legacy)
print("Synced extract_asr_colab.ipynb -> 04_Extract_ASR_Whisper.ipynb")
