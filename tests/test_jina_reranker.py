from types import SimpleNamespace
from urllib.error import URLError

from PIL import Image

from src.services import jina_reranker


def _settings(**overrides):
    values = {
        "jina_reranker_enabled": True,
        "jina_reranker_api_key": "test-key",
        "jina_reranker_base_url": "https://example.invalid/v1/rerank",
        "jina_reranker_model": "jina-reranker-m0",
        "jina_reranker_candidate_pool": 2,
        "jina_reranker_image_max_side": 256,
        "jina_reranker_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _results():
    return [
        {"rank": 1, "score": 0.38, "frame_path": "one.jpg", "video_id": "L21_V001"},
        {"rank": 2, "score": 0.37, "frame_path": "two.jpg", "video_id": "L21_V002"},
        {"rank": 3, "score": 0.36, "frame_path": "three.jpg", "video_id": "L21_V003"},
    ]


def test_jina_reranker_reorders_pool_and_preserves_retrieval_score(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), "red").save(image_path)
    sent = {}

    monkeypatch.setattr(jina_reranker, "resolve_keyframe_path", lambda item: image_path)

    def fake_post(payload, **_kwargs):
        sent.update(payload)
        return {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.64},
            ]
        }

    monkeypatch.setattr(jina_reranker, "_post_rerank", fake_post)

    ranked = jina_reranker.rerank_kis_results("red object", _results(), settings=_settings())

    assert sent["model"] == "jina-reranker-m0"
    assert len(sent["documents"]) == 2
    assert sent["documents"][0]["image"].startswith("data:image/jpeg;base64,")
    assert [item["frame_path"] for item in ranked] == ["two.jpg", "one.jpg", "three.jpg"]
    assert ranked[0]["score"] == 0.91
    assert ranked[0]["retrieval_score"] == 0.37
    assert [item["rank"] for item in ranked] == [1, 2, 3]


def test_jina_reranker_returns_original_results_when_disabled(monkeypatch):
    monkeypatch.setattr(jina_reranker, "_post_rerank", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
    results = _results()
    assert jina_reranker.rerank_kis_results("query", results, settings=_settings(jina_reranker_enabled=False)) is results


def test_jina_reranker_falls_back_when_api_fails(monkeypatch, tmp_path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), "blue").save(image_path)
    monkeypatch.setattr(jina_reranker, "resolve_keyframe_path", lambda item: image_path)
    monkeypatch.setattr(jina_reranker, "_post_rerank", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")))

    results = _results()
    assert jina_reranker.rerank_kis_results("query", results, settings=_settings()) is results
