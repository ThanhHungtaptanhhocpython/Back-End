import sys
sys.path.append('C:/Users/Lenovo/Documents/GitHub/AIC/Backend/Back-End')
from src.services.beit3_retriever import get_beit3_retriever
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv(dotenv_path='C:/Users/Lenovo/Documents/GitHub/AIC/Backend/Back-End/.env')

retriever = get_beit3_retriever()

queries = [
    "Một người đứng dưới nước và rọi đèn.",
    "cảnh người này kéo lưới cá lúc bình minh",
    "một nhóm người khác tiến đến dùng máy quay ghi hình"
]

results_by_event = []
for q in queries:
    res = retriever.search_visual(q, top_k=500)
    results_by_event.append(res)

print("Done searching.")

# Group by video
video_events = defaultdict(lambda: [[], [], []])

for e_idx, res_list in enumerate(results_by_event):
    for item in res_list:
        vid = item['video_id']
        fid = int(item['frame_id'])
        score = float(item['score'])
        video_events[vid][e_idx].append((fid, score))

best_match = None
best_score = -1

for vid, events in video_events.items():
    if not events[0] or not events[1] or not events[2]:
        continue
    
    # Sort frames for each event
    e1_list = sorted(events[0], key=lambda x: x[0])
    e2_list = sorted(events[1], key=lambda x: x[0])
    e3_list = sorted(events[2], key=lambda x: x[0])
    
    # Find valid chronological sequence E1 < E2 < E3
    for f1, s1 in e1_list:
        for f2, s2 in e2_list:
            if f2 <= f1: continue
            for f3, s3 in e3_list:
                if f3 <= f2: continue
                
                total_score = s1 + s2 + s3
                if total_score > best_score:
                    best_score = total_score
                    best_match = (vid, f1, f2, f3, s1, s2, s3)

if best_match:
    print(f"FOUND BEST SEQUENCE: Video {best_match[0]}")
    print(f"E1: Frame {best_match[1]} (Score {best_match[4]:.4f})")
    print(f"E2: Frame {best_match[2]} (Score {best_match[5]:.4f})")
    print(f"E3: Frame {best_match[3]} (Score {best_match[6]:.4f})")
    print(f"Total Score: {best_score:.4f}")
else:
    print("No sequence found in top 500.")
