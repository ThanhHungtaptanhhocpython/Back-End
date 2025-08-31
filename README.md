# Back-End

# File Structure
Cần tổ chức lại các thư mục dict và data
```
Back-End/
├── app.py                     # Entry point chính của ứng dụng
├── config.py                  # Cấu hình (DB, secret key, env variables)
├── requirements.txt           # List các package cần cài đặt
├── src/                       # Thư mục chính cho code
│   ├── __init__.py
│   ├── controllers/                # Chứa các route / endpoint
│   │   ├── __init__.py
│   │   ├── user_controller.py
│   ├── services/              # Chứa logic xử lý nghiệp vụ
│   │   ├── __init__.py
│   │   ├── user_service.py
│   ├── utils/                 # Các tiện ích helper, faiss, vlm, logging,...
│   │   ├── __init__.py
│   │   ├── faiss_processing.py
│   │   ├── vlm_processing.py
|   |   ├── combine_utils.py
│   │   └── nlp_processing.py
│   ├── data/Keyframes                 # Các tiện ích helper, faiss, vlm, logging,...
│   │   ├── dataset...
│   ├── dict/                 # Các tiện ích helper, faiss, vlm, logging,...
|   |   ├── faiss_index_clip.bin
│   │   └── metadata_clip.json
├── tests/                     # Chứa các test case
│   ├── __init__.py
│   ├── test_user.py
│   └── test_video.py
└── README.md
```
## Create venv (window users)

```
conda create --name AIChallenge2025
conda activate AIChallenge2025
```

## Set up

```
pip install -r requirements.txt
```

## Run Backend

```
python app.py
```
