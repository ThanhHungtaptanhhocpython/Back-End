import sys
sys.path.append('C:/Users/Lenovo/Documents/GitHub/AIC/Backend/Back-End')
from src.services.beit3_retriever import get_beit3_retriever
from dotenv import load_dotenv

load_dotenv(dotenv_path='C:/Users/Lenovo/Documents/GitHub/AIC/Backend/Back-End/.env')

retriever = get_beit3_retriever()

queries = [
    "Một người đứng dưới nước và rọi đèn.",
    "Tiếp theo là cảnh người này kéo lưới cá lúc bình minh",
    "sau đó được một nhóm người khác tiến đến dùng máy quay ghi hình."
]

for q in queries:
    print(f"\n--- Query: {q} ---")
    results = retriever.search_visual(q, top_k=5)
    for res in results:
        print(f"Video: {res['video_id']}, Frame: {res['frame_id']}, Score: {res['score']}")
