import os
import shutil
import sys
from pathlib import Path

from PIL import Image

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from src.config.settings import get_settings
from src.services import openrouter_vlm_verifier
from src.services.openrouter_vlm_verifier import verify_frames_with_openrouter_vlm


def test_openrouter_vlm_verifier_reranks_candidates_and_keeps_tail(monkeypatch):
    keyframes_root = Path("scratch") / "test_vlm_keyframes" / "Keyframes"
    image_dir = keyframes_root / "L1" / "L1_V001"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, color in (("000001.webp", "red"), ("000002.webp", "blue")):
        Image.new("RGB", (64, 64), color=color).save(image_dir / name)

    monkeypatch.setenv("AGENT_VLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("KEYFRAMES_ROOT", str(keyframes_root))
    monkeypatch.setenv("AGENT_VLM_MAX_CANDIDATES", "3")
    monkeypatch.setenv("AGENT_VLM_BATCH_SIZE", "3")
    monkeypatch.setenv("AGENT_VLM_MODEL", "google/gemini-2.5-flash")
    get_settings.cache_clear()

    def fake_request(messages, model, max_tokens, timeout):
        assert model == "google/gemini-2.5-flash"
        return {
            "items": [
                {"id": "c1", "score": 0.2, "decision": "wrong", "reason": "wrong scene", "missing": ["target"]},
                {"id": "c2", "score": 0.95, "decision": "match", "reason": "matches target", "matched": ["target"]},
            ]
        }

    monkeypatch.setattr(openrouter_vlm_verifier, "_request_openrouter_vlm", fake_request)
    frames = [
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/000001.webp", "verification_score": 0.9},
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/000002.webp", "verification_score": 0.4},
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/missing.webp", "verification_score": 0.8},
    ]
    plan = {"original_query": "target", "visual_query": "target", "must_have_checks": []}

    verified, summary = verify_frames_with_openrouter_vlm(frames, plan)

    assert summary["method"] == "openrouter_vlm"
    assert summary["evaluated"] == 2
    assert summary["missing_images"] == 1
    assert verified[0]["frame_path"].endswith("000002.webp")
    assert verified[0]["agent_verification"]["method"] == "openrouter_vlm"
    assert verified[0]["vlm_score"] == 0.95
    assert any(item["frame_path"].endswith("missing.webp") for item in verified)
    shutil.rmtree(keyframes_root.parent, ignore_errors=True)
    get_settings.cache_clear()