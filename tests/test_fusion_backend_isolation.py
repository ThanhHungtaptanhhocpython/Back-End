"""Fix 1 -- cross-backend fusion isolation.

`multimodal_search` must never join OCR/ASR evidence (whose `faiss_id` /
`nearest_faiss_id` were precomputed against the BEiT3 index) to an active
visual result from a *different* backend by raw integer id. A numeric
collision between a Jina `vector_id` and a BEiT3 ASR/OCR id must not be able
to attach unrelated text to a frame.
"""

from __future__ import annotations

import os
import sys

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import src.services.fusion_service as fusion_service  # noqa: E402
from src.services.fusion_service import merge_by_video_timestamp, multimodal_search  # noqa: E402


class _FakeElastic:
    def __init__(self, ocr=None, asr=None):
        self._ocr = ocr or []
        self._asr = asr or []

    def search_ocr(self, query, topk=100):
        return [dict(r) for r in self._ocr]

    def search_asr(self, query, topk=100):
        return [dict(r) for r in self._asr]


def _patch_common(monkeypatch, visual, ocr=None, asr=None):
    monkeypatch.setattr(
        "src.services.user_service.getImageDataSingleTextSearch",
        lambda q, k: [dict(r) for r in visual],
    )
    monkeypatch.setattr(
        fusion_service, "get_elastic_processor", lambda: _FakeElastic(ocr=ocr, asr=asr)
    )
    # keep reranking cheap + offline
    monkeypatch.setattr(
        fusion_service.QueryPlanner, "generate_vqa_question",
        staticmethod(lambda q: "Is there a scene?"), raising=False,
    )


# The number that collides across the two id spaces.
_COLLIDE = 4242


def _jina_visual():
    return [{
        "vector_id": _COLLIDE, "score": 0.91,
        "video_id": "L21_V001", "timestamp": 10.0,
        "frame_id": "kf_0002", "frame_idx": 17,
        "frame_path": "L21/L21_V001/kf_0002.jpg",
        "asset_key": "L21/L21_V001/kf_0002.jpg",
        "split": "L21", "retrieval_backend": "jina_clip_v2",
    }]


def _beit3_visual():
    row = _jina_visual()[0].copy()
    row["retrieval_backend"] = "beit3"
    return [row]


def test_colliding_beit3_asr_id_cannot_contaminate_a_jina_visual_result(monkeypatch):
    # ASR hit from a DIFFERENT video, far away in time, but whose
    # nearest_faiss_id happens to equal the Jina frame's vector_id.
    asr = [
        {"nearest_faiss_id": _COLLIDE, "video_id": "L30_V099",
         "start_time": 900.0, "end_time": 905.0, "text": "unrelated ambulance", "_score": 5.0},
        {"nearest_faiss_id": _COLLIDE + 1, "video_id": "L30_V099",
         "start_time": 400.0, "end_time": 402.0, "text": "noise", "_score": 1.0},
    ]
    _patch_common(monkeypatch, _jina_visual(), asr=asr)

    results = multimodal_search(
        "a man walking", "", "ambulance",
        {"visual": 0.5, "ocr": 0.0, "asr": 0.5}, topk=10, original_query="",
    )
    assert len(results) == 1
    top = results[0]
    assert top["vector_id"] == _COLLIDE
    assert top["score_breakdown"]["asr"] == 0.0
    assert "asr_text" not in top  # the unrelated transcript never attached


def test_beit3_visual_still_joins_its_own_asr_id(monkeypatch):
    # Same collision id, but now the active backend IS BEiT3, so
    # nearest_faiss_id == vector_id is a legitimate same-frame match.
    asr = [
        {"nearest_faiss_id": _COLLIDE, "video_id": "L21_V001",
         "start_time": 9.0, "end_time": 11.0, "text": "same frame transcript", "_score": 5.0},
        {"nearest_faiss_id": 99, "video_id": "L21_V001",
         "start_time": 1.0, "end_time": 2.0, "text": "other", "_score": 1.0},
    ]
    _patch_common(monkeypatch, _beit3_visual(), asr=asr)

    results = multimodal_search(
        "a man walking", "", "transcript",
        {"visual": 0.5, "ocr": 0.0, "asr": 0.5}, topk=10, original_query="",
    )
    joined = [r for r in results if r.get("faiss_id") == _COLLIDE]
    assert joined and joined[0]["asr_text"] == "same frame transcript"


def test_jina_ocr_asr_fuse_only_on_matching_video_and_timestamp(monkeypatch):
    ocr = [
        {"video_id": "L21_V001", "timestamp": 10.4, "ocr_text": "HOSPITAL", "_score": 9.0},
        {"video_id": "L21_V001", "timestamp": 55.0, "ocr_text": "far away", "_score": 1.0},
    ]
    _patch_common(monkeypatch, _jina_visual(), ocr=ocr)

    results = multimodal_search(
        "a man walking", "hospital sign", "",
        {"visual": 0.5, "ocr": 0.5, "asr": 0.0}, topk=10, original_query="",
    )
    assert len(results) == 1
    top = results[0]
    assert top["score_breakdown"]["ocr"] > 0.0        # the 10.4s OCR hit fused
    assert top["ocr_text"] == "HOSPITAL"


def test_merge_by_video_timestamp_never_creates_a_row_from_text():
    visual = [{"vector_id": 1, "video_id": "V1", "timestamp": 3.0, "normalized_score": 1.0}]
    ocr = [{"video_id": "V2", "timestamp": 3.0, "ocr_text": "x", "normalized_score": 1.0}]
    asr = [{"video_id": "V2", "nearest_timestamp": 3.0, "text": "y", "normalized_score": 1.0}]
    out = merge_by_video_timestamp(visual, ocr, asr, {"visual": 1.0, "ocr": 1.0, "asr": 1.0})
    assert [d["vector_id"] for d in out] == [1]
    assert out[0]["score_breakdown"]["ocr"] == 0.0
    assert out[0]["score_breakdown"]["asr"] == 0.0


def test_merge_by_video_timestamp_drops_untimestamped_text():
    visual = [{"vector_id": 1, "video_id": "V1", "timestamp": 3.0, "normalized_score": 1.0}]
    ocr = [{"video_id": "V1", "ocr_text": "no timestamp here", "normalized_score": 1.0}]
    out = merge_by_video_timestamp(visual, ocr, [], {"visual": 1.0, "ocr": 1.0, "asr": 0.0})
    assert out[0]["score_breakdown"]["ocr"] == 0.0
    assert "ocr_text" not in out[0]
