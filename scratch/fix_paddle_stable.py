import json
import os

def fix_paddle_stable_version():
    base_dir = r"c:\Users\Lenovo\Documents\GitHub\AIC\Backend\Back-End"
    nb_files = [
        os.path.join(base_dir, "scripts", "notebooks", "extract_ocr_colab.ipynb"),
        os.path.join(base_dir, "scripts", "notebooks", "03_Extract_OCR_PaddleOCR.ipynb")
    ]
    
    for nbf in nb_files:
        if os.path.exists(nbf):
            with open(nbf, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    # Fix pip cell
                    if "pip install" in src and "paddleocr" in src:
                        new_pip = """!pip install -q "paddleocr>=2.7.3,<2.8.0" paddlepaddle-gpu tqdm Pillow

import paddle
print(f"✅ PaddlePaddle GPU Sẵn sàng: {paddle.is_compiled_with_cuda()}")
if paddle.is_compiled_with_cuda():
    print(f"⚡ Thiết bị GPU: {paddle.device.get_device()}")
    print("🚀 TUYỆT VỜI! Đã kích hoạt PaddleOCR PP-OCRv4 trên GPU A100!")"""
                        cell["source"] = [line + "\n" for line in new_pip.split("\n")[:-1]] + [new_pip.split("\n")[-1]]
                        
                    # Fix init cell
                    if "from paddleocr import PaddleOCR" in src:
                        new_init = """from paddleocr import PaddleOCR
import logging

logging.getLogger('ppocr').setLevel(logging.ERROR)

print("Đang nạp mô hình PaddleOCR (PP-OCRv4 Tiếng Việt) lên GPU A100...")

# Khởi tạo PaddleOCR chuẩn ổn định 100%
ocr = PaddleOCR(
    use_angle_cls=True,   # Tự động xoay chữ nghiêng/ngang
    lang='vi',            # Tiếng Việt có dấu
    use_gpu=True,         # Chạy trên GPU A100
    show_log=False
)

print("✅ PaddleOCR PP-OCRv4 đã sẵn sàng hoạt động trên GPU A100!")"""
                        cell["source"] = [line + "\n" for line in new_init.split("\n")[:-1]] + [new_init.split("\n")[-1]]
                        
            with open(nbf, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Pinned stable PaddleOCR version in {os.path.basename(nbf)}")

fix_paddle_stable_version()
