import json
import os

def update_asr_map_paths():
    base_dir = r"c:\Users\Lenovo\Documents\GitHub\AIC\Backend\Back-End"
    nb_files = [
        os.path.join(base_dir, "scripts", "notebooks", "extract_asr_colab.ipynb"),
        os.path.join(base_dir, "scripts", "notebooks", "04_Extract_ASR_Whisper.ipynb")
    ]
    
    for nbf in nb_files:
        if os.path.exists(nbf):
            with open(nbf, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    if "map_zip_candidates" in src and "AIC 2025 1" not in src:
                        old_code = 'map_zip_candidates = [\n    Path("/content/drive/MyDrive/map-keyframes.zip"),\n    Path("/content/drive/MyDrive/AIC2025/map-keyframes.zip"),\n    Path("/content/map-keyframes.zip")\n]'
                        new_code = 'map_zip_candidates = [\n    Path("/content/drive/MyDrive/AIC 2025 1/map-keyframes.zip"),\n    Path("/content/drive/MyDrive/AIC 2025/map-keyframes.zip"),\n    Path("/content/drive/MyDrive/AIC2025/map-keyframes.zip"),\n    Path("/content/drive/MyDrive/map-keyframes.zip"),\n    Path("/content/map-keyframes.zip")\n]'
                        src = src.replace(old_code, new_code)
                        cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + ([src.split("\n")[-1]] if src.split("\n")[-1] else [])
                        
            with open(nbf, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Updated {os.path.basename(nbf)}")

update_asr_map_paths()
