"""Router tests for the video playback / frame-capture endpoints.

They assert the response envelope, status codes, and — importantly — that
hitting these routes never imports or initialises the BEiT-3 search stack.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from src.services import video_frame_preview_service as vfps
from src.services import video_playback_service as vps

client = TestClient(app)


def _make_zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


@pytest.fixture(autouse=True)
def fixture_service(tmp_path: Path, monkeypatch):
    media = _make_zip(
        tmp_path / "media-info.zip",
        {"media-info/L21_V001.json": json.dumps(
            {"title": "sample", "watch_url": "https://youtube.com/watch?v=abc123", "length": 1262}
        )},
    )
    kmap = _make_zip(
        tmp_path / "map-keyframes.zip",
        {"map-keyframes/L21_V001.csv": "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,11.7333,30.0,351\n"},
    )
    service = vps.VideoPlaybackService(media, kmap)
    monkeypatch.setattr(vps, "get_video_playback_service", lambda: service)

    # Route tests must not reach out to YouTube; force the "no preview" path.
    class _StubPreview:
        def get_or_create(self, **_kwargs):
            raise vfps.FramePreviewError("FFmpeg binary is not available on this server.")

        def get_existing(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: _StubPreview())
    return service


def test_playback_without_frame_idx():
    resp = client.get("/users/videos/L21_V001/playback")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total_items"] == 1
    item = body["data"]["items"][0]
    assert item["video_id"] == "L21_V001"
    assert item["watch_url"].endswith("abc123")
    assert item["fps"] == 30.0
    assert item["duration_seconds"] == 1262
    assert item["playback_offset_seconds"] == 0.0
    assert item["start_seconds"] is None


def test_playback_with_frame_idx_returns_start_time():
    resp = client.get("/users/videos/L21_V001/playback", params={"frame_idx": 351})
    assert resp.status_code == 200
    item = resp.json()["data"]["items"][0]
    assert item["frame_idx"] == 351
    assert item["start_seconds"] == pytest.approx(11.7, abs=1e-6)


def test_playback_root_prefix_alias_works():
    resp = client.get("/videos/L21_V001/playback")
    assert resp.status_code == 200
    assert resp.json()["data"]["items"][0]["video_id"] == "L21_V001"


def test_playback_unknown_video_is_404():
    resp = client.get("/users/videos/L99_V999/playback")
    assert resp.status_code == 404
    assert resp.json()["success"] is False


def test_capture_returns_frame_idx():
    resp = client.post(
        "/users/videos/L21_V001/capture",
        json={"playback_time_seconds": 11.7333},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["video_id"] == "L21_V001"
    assert data["frame_idx"] == 351
    assert data["fps"] == 30.0
    assert data["source_time_seconds"] == pytest.approx(11.7333)


def test_capture_response_carries_preview_contract_on_extractor_failure():
    resp = client.post(
        "/users/videos/L21_V001/capture",
        json={"playback_time_seconds": 11.7333},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    # The valid frame index is still returned...
    assert data["frame_idx"] == 351
    # ...but no original review image is substituted.
    assert data["preview_url"] is None
    assert "FFmpeg" in (data["preview_error"] or "")


def test_captured_frame_route_serves_existing_still_and_404s_otherwise(tmp_path, monkeypatch):
    still = tmp_path / "L21_V001" / "351.webp"
    still.parent.mkdir(parents=True)
    still.write_bytes(b"RIFF\x00\x00\x00\x00WEBPfake")

    class _Preview:
        def get_existing(self, video_id, frame_idx):
            path = tmp_path / str(video_id) / f"{int(frame_idx)}.webp"
            return path if path.is_file() else None

        def get_or_create(self, **_kwargs):
            raise vfps.FramePreviewError("unused")

    monkeypatch.setattr(vfps, "get_video_frame_preview_service", lambda: _Preview())

    ok = client.get("/users/videos/captures/L21_V001/351.webp")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/webp"

    missing = client.get("/users/videos/captures/L21_V001/999.webp")
    assert missing.status_code == 404


def test_capture_negative_timestamp_is_400():
    resp = client.post(
        "/users/videos/L21_V001/capture",
        json={"playback_time_seconds": -3},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_capture_beyond_duration_is_400():
    resp = client.post(
        "/users/videos/L21_V001/capture",
        json={"playback_time_seconds": 99999},
    )
    assert resp.status_code == 400


def test_capture_missing_body_is_422():
    resp = client.post("/users/videos/L21_V001/capture", json={})
    assert resp.status_code in (400, 422)


def test_endpoints_do_not_load_beit3():
    for module_name in ("src.services.beit3_retriever", "torch"):
        sys.modules.pop(module_name, None)
    client.get("/users/videos/L21_V001/playback", params={"frame_idx": 10})
    client.post("/users/videos/L21_V001/capture", json={"playback_time_seconds": 5})
    assert "src.services.beit3_retriever" not in sys.modules
