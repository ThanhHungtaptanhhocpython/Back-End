import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import kagglehub

# -----------------------------
# Download dataset
# -----------------------------
path_frame = kagglehub.dataset_download("trietdeptrai/frames2go")
print("Path to frames2go files:", path_frame)

# -----------------------------
# Parameters
# -----------------------------
output_root = "frames2go_4kb"
target_kb = 4        # target dung lượng
max_error_kb = 10    # sai số cho phép
max_workers = 16      # số thread
resize_size = (320, 320)  # resize để dễ đạt dung lượng
os.makedirs(output_root, exist_ok=True)

# -----------------------------
# Function nén ảnh
# -----------------------------
def compress_near_target(img_path, out_path, target_kb=4, max_error_kb=10):
    target_size = target_kb * 1024
    max_size = (target_kb + max_error_kb) * 1024

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(resize_size, Image.LANCZOS)
        quality = 25  # bắt đầu quality cố định
        img.save(out_path, "WEBP", quality=quality, method=6)

        size = os.path.getsize(out_path)
        # Nếu ảnh vẫn quá lớn → giảm quality xuống thấp hơn
        if size > max_size:
            quality = 15
            img.save(out_path, "WEBP", quality=quality, method=6)

        return out_path, os.path.getsize(out_path)//1024
    except Exception as e:
        return img_path, str(e)

# -----------------------------
# Multi-thread processing
# -----------------------------
tasks = []
from tqdm  import tqdm 
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for part in tqdm(sorted(os.listdir(path_frame))):
        part_path = os.path.join(path_frame, part)
        if not os.path.isdir(part_path):
            continue
        for vid in tqdm(sorted(os.listdir(part_path))):
            vid_path = os.path.join(part_path, vid)
            if not os.path.isdir(vid_path):
                continue
            save_path = os.path.join(output_root, part, vid)
            os.makedirs(save_path, exist_ok=True)
            for item in os.listdir(vid_path):
                if item.endswith(".webp"):
                    in_file = os.path.join(vid_path, item)
                    out_file = os.path.join(save_path, item)
                    tasks.append(executor.submit(compress_near_target, in_file, out_file, target_kb, max_error_kb))

    for future in tqdm(as_completed(tasks), total=len(tasks)):
        out_file, size_kb = future.result()
#        print(f"{out_file}: {size_kb} KB")
