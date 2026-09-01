"""Unit tests for the exact captured-frame preview service.

None of these reach the network: the yt-dlp / FFmpeg seams are stubbed so we
exercise caching, LRU eviction, and graceful degradation only.
"""

from __future__ import annotations

import builtins
import os
import subprocess
from pathlib import Path

import pytest

from src.services import video_frame_preview_service as vfps
from src.services.video_frame_preview_service import (
    FramePreviewError,
    VideoFramePreviewService,
)

_WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 fake-frame-bytes"


def _wire_fake_tools(service: VideoFramePreviewService, monkeypatch, *, calls: list) -> None:
    """Make ``_extract`` succeed without touching yt-dlp / FFmpeg / the network."""
    monkeypatch.setattr(service, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(service, "_resolve_ytdlp", lambda: ["yt-dlp"])
    monkeypatch.setattr(
        service,
        "_resolve_media_url",
        lambda _cmd, _url: "http://stream.example/video.mp4",
    )

    def fake_run(cmd, *, what):
        calls.append(what)
        if what == "FFmpeg":
            Path(cmd[-1]).write_bytes(_WEBP)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(service, "_run", fake_run)


def test_extraction_writes_webp_and_returns_cache_key(tmp_path: Path, monkeypatch):
    service = VideoFramePreviewService(cache_dir=tmp_path)
    calls: list = []
    _wire_fake_tools(service, monkeypatch, calls=calls)

    key = service.get_or_create(
        video_id="L21_V001",
        frame_idx=351,
        watch_url="https://youtube.com/watch?v=abc123",
        target_seconds=11.7,
    )

    assert key == "L21_V001/351.webp"
    still = tmp_path / "L21_V001" / "351.webp"
    assert still.read_bytes() == _WEBP
    assert calls == ["FFmpeg"]  # one extraction


def test_cache_hit_performs_no_second_external_fetch(tmp_path: Path, monkeypatch):
    service = VideoFramePreviewService(cache_dir=tmp_path)
    calls: list = []
    _wire_fake_tools(service, monkeypatch, calls=calls)

    first = service.get_or_create(
        video_id="L21_V001", frame_idx=351,
        watch_url="https://youtube.com/watch?v=abc123", target_seconds=11.7,
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("cache hit must not re-run extraction")

    monkeypatch.setattr(service, "_extract", _boom)
    second = service.get_or_create(
        video_id="L21_V001", frame_idx=351,
        watch_url="https://youtube.com/watch?v=abc123", target_seconds=11.7,
    )

    assert first == second == "L21_V001/351.webp"
    assert calls == ["FFmpeg"]


def test_lru_eviction_keeps_cache_at_or_below_max_bytes(tmp_path: Path):
    entry_size = len(_WEBP)
    service = VideoFramePreviewService(cache_dir=tmp_path, cache_max_bytes=entry_size * 2)

    made: list[Path] = []
    for index, name in enumerate(("100.webp", "200.webp", "300.webp")):
        path = tmp_path / "L21_V001" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_WEBP)
        # Strictly increasing mtimes: 100 is oldest / least-recently-used.
        os.utime(path, (1_000 + index * 10, 1_000 + index * 10))
        made.append(path)

    service._enforce_limit()

    assert not made[0].exists()  # evicted
    assert made[1].exists()
    assert made[2].exists()
    total = sum(p.stat().st_size for p in tmp_path.rglob("*.webp"))
    assert total <= entry_size * 2


def test_missing_ffmpeg_degrades_gracefully(tmp_path: Path):
    service = VideoFramePreviewService(
        cache_dir=tmp_path, ffmpeg_bin="definitely-not-a-real-binary-xyz"
    )
    with pytest.raises(FramePreviewError, match="FFmpeg"):
        service.get_or_create(
            video_id="L21_V001", frame_idx=351,
            watch_url="https://youtube.com/watch?v=abc123", target_seconds=11.7,
        )
    assert not (tmp_path / "L21_V001" / "351.webp").exists()


def test_missing_ytdlp_degrades_gracefully(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vfps.shutil, "which", lambda _bin: None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yt_dlp":
            raise ImportError("no yt_dlp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    service = VideoFramePreviewService(cache_dir=tmp_path, ytdlp_bin="nope-not-real")
    with pytest.raises(FramePreviewError, match="yt-dlp"):
        service._resolve_ytdlp()


def test_ffmpeg_failure_leaves_no_partial_still(tmp_path: Path, monkeypatch):
    service = VideoFramePreviewService(cache_dir=tmp_path)
    monkeypatch.setattr(service, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(service, "_resolve_ytdlp", lambda: ["yt-dlp"])
    monkeypatch.setattr(service, "_resolve_media_url", lambda _c, _u: "http://s/v.mp4")
    monkeypatch.setattr(
        service,
        "_run",
        lambda cmd, *, what: subprocess.CompletedProcess(cmd, 1, "", "moov atom not found"),
    )

    with pytest.raises(FramePreviewError, match="decode the frame"):
        service.get_or_create(
            video_id="L21_V001", frame_idx=351,
            watch_url="https://youtube.com/watch?v=abc123", target_seconds=11.7,
        )
    assert not (tmp_path / "L21_V001" / "351.webp").exists()


def test_blank_watch_url_is_reported_not_raised_as_generic(tmp_path: Path):
    service = VideoFramePreviewService(cache_dir=tmp_path)
    with pytest.raises(FramePreviewError, match="No source video URL"):
        service.get_or_create(
            video_id="L21_V001", frame_idx=1, watch_url="  ", target_seconds=0.0,
        )


def test_get_existing_bumps_recency(tmp_path: Path):
    service = VideoFramePreviewService(cache_dir=tmp_path)
    still = tmp_path / "L21_V001" / "351.webp"
    still.parent.mkdir(parents=True)
    still.write_bytes(_WEBP)
    os.utime(still, (1_000, 1_000))

    found = service.get_existing("L21_V001", 351)
    assert found == still
    assert still.stat().st_mtime > 1_000  # touched
