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
    monkeypatch.setenv("AGENT_VLM_CACHE_ENABLED", "false")
    get_settings.cache_clear()

    def fake_request(messages, model, max_tokens, timeout):
        assert model == "google/gemini-2.5-flash"
        return {
            "items": [
                {"id": "c1", "score": 0.2, "decision": "wrong", "reason": "wrong scene", "matched": [], "missing": ["target"]},
                {"id": "c2", "score": 0.95, "decision": "match", "reason": "matches target", "matched": ["target"], "missing": []},
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


def test_openrouter_vlm_verifier_diversifies_candidate_pool_by_video(monkeypatch):
    keyframes_root = Path("scratch") / "test_vlm_diverse_keyframes" / "Keyframes"
    for video_id, name, color in (
        ("L1_V001", "000001.webp", "red"),
        ("L1_V001", "000002.webp", "green"),
        ("L1_V002", "000001.webp", "blue"),
    ):
        image_dir = keyframes_root / "L1" / video_id
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=color).save(image_dir / name)

    monkeypatch.setenv("AGENT_VLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("KEYFRAMES_ROOT", str(keyframes_root))
    monkeypatch.setenv("AGENT_VLM_MAX_CANDIDATES", "2")
    monkeypatch.setenv("AGENT_VLM_CANDIDATE_POOL", "3")
    monkeypatch.setenv("AGENT_VLM_PER_VIDEO_LIMIT", "1")
    monkeypatch.setenv("AGENT_VLM_BATCH_SIZE", "2")
    monkeypatch.setenv("AGENT_VLM_CACHE_ENABLED", "false")
    get_settings.cache_clear()

    captured_text = []

    def fake_request(messages, model, max_tokens, timeout):
        text = messages[1]["content"][0]["text"]
        captured_text.append(text)
        return {
            "items": [
                {"id": "c1", "score": 0.3, "decision": "partial", "reason": "same video candidate", "matched": [], "missing": ["target"]},
                {"id": "c2", "score": 0.9, "decision": "match", "reason": "diverse candidate", "matched": ["target"], "missing": []},
            ]
        }

    monkeypatch.setattr(openrouter_vlm_verifier, "_request_openrouter_vlm", fake_request)
    frames = [
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/000001.webp", "verification_score": 0.99},
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/000002.webp", "verification_score": 0.98},
        {"video_id": "L1_V002", "frame_path": "L1/L1_V002/000001.webp", "verification_score": 0.50},
    ]

    verified, summary = verify_frames_with_openrouter_vlm(frames, {"original_query": "target", "visual_query": "target"})

    assert summary["candidate_pool"] == 3
    assert summary["per_video_limit"] == 1
    assert "video=L1_V002" in "\n".join(captured_text)
    assert verified[0]["video_id"] == "L1_V002"
    shutil.rmtree(keyframes_root.parent, ignore_errors=True)
    get_settings.cache_clear()


def test_openrouter_vlm_verifier_reuses_persistent_verdict_cache(monkeypatch):
    scratch_root = Path("scratch") / "test_vlm_verdict_cache"
    keyframes_root = scratch_root / "Keyframes"
    image_dir = keyframes_root / "L1" / "L1_V001"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="yellow").save(image_dir / "000001.webp")

    monkeypatch.setenv("AGENT_VLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("KEYFRAMES_ROOT", str(keyframes_root))
    monkeypatch.setenv("AGENT_VLM_MAX_CANDIDATES", "1")
    monkeypatch.setenv("AGENT_VLM_CACHE_ENABLED", "true")
    monkeypatch.setenv("AGENT_VLM_CACHE_PATH", str(scratch_root / "cache.json"))
    monkeypatch.setenv("AGENT_VLM_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()

    calls = []

    def fake_request(messages, model, max_tokens, timeout):
        calls.append(messages)
        return {
            "items": [
                {
                    "id": "c1",
                    "score": 0.92,
                    "decision": "match",
                    "reason": "visible target",
                    "matched": ["yellow subject"],
                    "missing": [],
                }
            ]
        }

    monkeypatch.setattr(openrouter_vlm_verifier, "_request_openrouter_vlm", fake_request)
    frames = [{"video_id": "L1_V001", "frame_path": "L1/L1_V001/000001.webp", "verification_score": 0.5}]
    plan = {"original_query": "yellow subject", "visual_query": "a yellow subject"}

    first, first_summary = verify_frames_with_openrouter_vlm(frames, plan)
    second, second_summary = verify_frames_with_openrouter_vlm(frames, plan)

    assert len(calls) == 1
    assert first_summary["cache_hits"] == 0
    assert second_summary["cache_hits"] == 1
    assert second_summary["api_calls"] == 0
    assert first[0]["agent_verification"]["vlm_source"] == "api"
    assert second[0]["agent_verification"]["vlm_source"] == "cache"
    shutil.rmtree(scratch_root, ignore_errors=True)
    get_settings.cache_clear()


def test_openrouter_vlm_verifier_rejects_invalid_contract_items(monkeypatch):
    scratch_root = Path("scratch") / "test_vlm_contract"
    keyframes_root = scratch_root / "Keyframes"
    image_dir = keyframes_root / "L1" / "L1_V001"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, color in (("000001.webp", "red"), ("000002.webp", "blue")):
        Image.new("RGB", (64, 64), color=color).save(image_dir / name)

    monkeypatch.setenv("AGENT_VLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("KEYFRAMES_ROOT", str(keyframes_root))
    monkeypatch.setenv("AGENT_VLM_MAX_CANDIDATES", "2")
    monkeypatch.setenv("AGENT_VLM_BATCH_SIZE", "2")
    monkeypatch.setenv("AGENT_VLM_CACHE_ENABLED", "false")
    get_settings.cache_clear()

    def fake_request(messages, model, max_tokens, timeout):
        return {
            "items": [
                {"id": "c1", "score": 0.8, "decision": "match", "reason": "valid", "matched": ["target"], "missing": []},
                {"id": "c2", "score": 1.0, "decision": "perfect", "reason": "invalid enum", "matched": [], "missing": []},
            ]
        }

    monkeypatch.setattr(openrouter_vlm_verifier, "_request_openrouter_vlm", fake_request)
    frames = [
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/000001.webp", "verification_score": 0.7},
        {"video_id": "L1_V001", "frame_path": "L1/L1_V001/000002.webp", "verification_score": 0.6},
    ]

    verified, summary = verify_frames_with_openrouter_vlm(frames, {"original_query": "target", "visual_query": "target"})

    assert summary["status"] == "partial"
    assert summary["fallback_used"] is True
    assert summary["evaluated"] == 1
    assert any("invalid decision for c2" in error for error in summary["contract_errors"])
    assert verified[0]["agent_verification"]["vlm_contract_version"] == "agent-vlm-verdict-v2"
    shutil.rmtree(scratch_root, ignore_errors=True)
    get_settings.cache_clear()


def test_openrouter_vlm_verifier_retries_then_falls_back(monkeypatch):
    scratch_root = Path("scratch") / "test_vlm_retry"
    keyframes_root = scratch_root / "Keyframes"
    image_dir = keyframes_root / "L1" / "L1_V001"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="green").save(image_dir / "000001.webp")

    monkeypatch.setenv("AGENT_VLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setenv("KEYFRAMES_ROOT", str(keyframes_root))
    monkeypatch.setenv("AGENT_VLM_MAX_CANDIDATES", "1")
    monkeypatch.setenv("AGENT_VLM_MAX_RETRIES", "1")
    monkeypatch.setenv("AGENT_VLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("AGENT_VLM_CACHE_ENABLED", "false")
    get_settings.cache_clear()

    calls = []

    def failing_request(messages, model, max_tokens, timeout):
        calls.append(messages)
        raise TimeoutError("OpenRouter timed out")

    monkeypatch.setattr(openrouter_vlm_verifier, "_request_openrouter_vlm", failing_request)
    frames = [{"video_id": "L1_V001", "frame_path": "L1/L1_V001/000001.webp", "verification_score": 0.7}]

    verified, summary = verify_frames_with_openrouter_vlm(frames, {"original_query": "target", "visual_query": "target"})

    assert len(calls) == 2
    assert verified == frames
    assert summary["status"] == "fallback"
    assert summary["fallback_used"] is True
    assert summary["api_calls"] == 2
    assert summary["retries"] == 1
    shutil.rmtree(scratch_root, ignore_errors=True)
    get_settings.cache_clear()
