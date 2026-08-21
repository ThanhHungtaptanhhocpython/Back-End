from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_image_search_by_faiss_index():
    response = client.post(
        "/users/imagesearch",
        data={
            "faiss_index": "100",
            "topk": "5"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]["items"]) == 5
    assert data["data"]["items"][0]["vector_id"] == 100
