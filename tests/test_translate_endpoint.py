from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from src.utils.nlp_processing import Translation

client = TestClient(app)

@patch("src.utils.nlp_processing.GoogleTranslator")
def test_translate_vi_to_en(mock_google):
    src = "thu hoạch dứa"
    mock_google.return_value.translate.return_value = "harvesting pineapples"
    Translation._cache.pop(("vi", "en", src), None)
    response = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["from_lang"] == "vi"
    assert data["to_lang"] == "en"
    assert "pineapple" in data["translated_text"].lower()
    assert data["provider"] == "google"
    assert data["translated"] is True

@patch("src.utils.nlp_processing.GoogleTranslator")
def test_translate_root_alias(mock_google):
    src = "người phụ nữ mặc áo hồng"
    mock_google.return_value.translate.return_value = "a woman wearing a pink shirt"
    Translation._cache.pop(("vi", "en", src), None)
    response = client.post(
        "/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "pink" in data["translated_text"].lower()


@patch("src.utils.nlp_processing.Translation._translate_with_openrouter")
@patch("src.utils.nlp_processing.GoogleTranslator")
def test_unaccented_vietnamese_prefers_contextual_translator(mock_google, mock_openrouter):
    src = "nguoi dan ong dung duoi nuoc"
    mock_openrouter.return_value = "a man standing in the water"
    Translation._cache.pop(("vi", "en", src), None)

    response = client.post(
        "/users/translate",
        json={"text": src, "from_lang": "vi", "to_lang": "en"},
    )

    assert response.status_code == 200
    assert response.json()["translated_text"] == "a man standing in the water"
    assert response.json()["provider"] == "openrouter"
    mock_openrouter.assert_called_once_with(src)
    mock_google.return_value.translate.assert_not_called()


def test_translate_rejects_unsupported_language():
    response = client.post(
        "/users/translate",
        json={"text": "bonjour", "from_lang": "fr", "to_lang": "vi"},
    )
    assert response.status_code == 400
