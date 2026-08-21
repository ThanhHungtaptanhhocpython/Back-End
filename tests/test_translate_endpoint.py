import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_translate_vi_to_en():
    response = client.post(
        "/users/translate",
        json={
            "text": "thu hoạch dứa",
            "from_lang": "vi",
            "to_lang": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["from_lang"] == "vi"
    assert data["to_lang"] == "en"
    assert "pineapple" in data["translated_text"].lower()

def test_translate_root_alias():
    response = client.post(
        "/translate",
        json={
            "text": "người phụ nữ mặc áo hồng",
            "from_lang": "vi",
            "to_lang": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "pink" in data["translated_text"].lower()
