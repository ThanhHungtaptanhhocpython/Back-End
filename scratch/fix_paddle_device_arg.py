import json
import os

def fix_paddle_device_arg():
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
                    if "ocr = PaddleOCR(" in src:
                        new_init_code = """from paddleocr import PaddleOCR
import logging

# Tắt log rác
logging.getLogger('ppocr').setLevel(logging.ERROR)

print("Đang nạp mô hình PaddleOCR (PP-OCRv4 Tiếng Việt) lên GPU A100...")

# Khởi tạo PaddleOCR (Tự động kích hoạt GPU và tối ưu đa luồng)
try:
    ocr = PaddleOCR(lang='vi', use_textline_orientation=True)
except Exception:
    try:
        ocr = PaddleOCR(lang='vi', use_angle_cls=True)
    except Exception:
        ocr = PaddleOCR(lang='vi')

print("✅ PaddleOCR PP-OCRv4 đã sẵn sàng hoạt động trên GPU A100!")"""
                        cell["source"] = [line + "\n" for line in new_init_code.split("\n")[:-1]] + [new_init_code.split("\n")[-1]]
                        
            with open(nbf, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Updated PaddleOCR initialization in {os.path.basename(nbf)}")

fix_paddle_device_arg()
