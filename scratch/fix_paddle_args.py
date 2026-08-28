import json
import os

def fix_paddleocr_args():
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
                    if "show_log=False" in src or "use_angle_cls=True" in src:
                        old_code = "ocr = PaddleOCR(\n    use_angle_cls=True,   # Tự động xoay chữ nghiêng/ngang\n    lang='vi',            # Tiếng Việt có dấu\n    use_gpu=True,         # Chạy trên GPU A100\n    show_log=False\n)"
                        new_code = "ocr = PaddleOCR(\n    use_textline_orientation=True,  # Tự động nhận diện hướng chữ\n    lang='vi',                      # Tiếng Việt\n    use_gpu=True                    # Kích hoạt GPU A100\n)"
                        src = src.replace(old_code, new_code)
                        # Also replace any other show_log
                        src = src.replace(",\n    show_log=False", "")
                        src = src.replace("show_log=False", "")
                        cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + ([src.split("\n")[-1]] if src.split("\n")[-1] else [])
                        
            with open(nbf, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Fixed PaddleOCR arguments in {os.path.basename(nbf)}")

fix_paddleocr_args()
