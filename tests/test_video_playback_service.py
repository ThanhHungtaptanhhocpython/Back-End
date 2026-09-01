"""Unit tests for the BEiT-3-independent video playback / capture service."""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.services.video_playback_service import (
    VideoMetadataError,
    VideoNotFoundError,
    VideoPlaybackService,
    VideoRequestError,
)


def _media_info_json(watch_url: str = "https://youtube.com/watch?v=abc123", length: int = 1262) -> str:
    payload = {"title": "sample", "watch_url": watch_url, "length": length}
    return json.dumps(payload)


def _map_keyframes_csv(fps: float) -> str:
    # n, pts_time, fps, frame_idx — pts_time * fps == frame_idx in the dataset.
    rows = [
        "n,pts_time,fps,frame_idx",
        f"1,0.0,{fps},0",
        f"2,3.0,{fps},{math.floor(3.0 * fps)}",
        f"3,11.7333,{fps},{math.floor(11.7333 * fps + 1e-6)}",
    ]
    return "\n".join(rows) + "\n"


def _make_zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


@pytest.fixture
def flat_media_info(tmp_path: Path) -> Path:
    return _make_zip(
        tmp_path / "media-info-flat.zip",
        {"L21_V001.json": _media_info_json()},
    )


@pytest.fixture
def wrapped_media_info(tmp_path: Path) -> Path:
    return _make_zip(
        tmp_path / "media-info-wrapped.zip",
        {"media-info/L21_V001.json": _media_info_json()},
    )


@pytest.fixture
def flat_map_keyframes(tmp_path: Path) -> Path:
    return _make_zip(
        tmp_path / "map-keyframes-flat.zip",
        {"L21_V001.csv": _map_keyframes_csv(30.0)},
    )


@pytest.fixture
def wrapped_map_keyframes(tmp_path: Path) -> Path:
    return _make_zip(
        tmp_path / "map-keyframes-wrapped.zip",
        {"map-keyframes/L21_V001.csv": _map_keyframes_csv(30.0)},
    )


def test_reads_flat_archive_entries(flat_media_info: Path, flat_map_keyframes: Path):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    meta = service.get_metadata("L21_V001")
    assert meta.watch_url == "https://youtube.com/watch?v=abc123"
    assert meta.fps == 30.0
    assert meta.duration_seconds == 1262
    assert meta.playback_offset_seconds == 0.0


def test_reads_wrapped_archive_entries(wrapped_media_info: Path, wrapped_map_keyframes: Path):
    service = VideoPlaybackService(wrapped_media_info, wrapped_map_keyframes)
    meta = service.get_metadata("L21_V001")
    assert meta.fps == 30.0
    assert meta.watch_url.endswith("abc123")


def test_reads_directory_source(tmp_path: Path):
    media_dir = tmp_path / "media-info"
    map_dir = tmp_path / "map-keyframes"
    media_dir.mkdir()
    map_dir.mkdir()
    (media_dir / "L21_V001.json").write_text(_media_info_json(), encoding="utf-8")
    (map_dir / "L21_V001.csv").write_text(_map_keyframes_csv(25.0), encoding="utf-8")

    service = VideoPlaybackService(media_dir, map_dir)
    assert service.get_metadata("L21_V001").fps == 25.0


@pytest.mark.parametrize("fps", [29.97, 26.44])
def test_preserves_fractional_fps(tmp_path: Path, fps: float):
    media = _make_zip(tmp_path / f"mi-{fps}.zip", {"L21_V001.json": _media_info_json()})
    kmap = _make_zip(tmp_path / f"mk-{fps}.zip", {"L21_V001.csv": _map_keyframes_csv(fps)})
    service = VideoPlaybackService(media, kmap)
    assert service.get_metadata("L21_V001").fps == fps


def test_capture_milestone_30fps(flat_media_info: Path, flat_map_keyframes: Path):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    result = service.capture("L21_V001", 11.7333)
    assert result.frame_idx == 351
    assert result.fps == 30.0
    assert result.source_time_seconds == pytest.approx(11.7333)


def test_capture_milestone_2644fps(tmp_path: Path):
    media = _make_zip(tmp_path / "mi.zip", {"L21_V001.json": _media_info_json(length=2000)})
    kmap = _make_zip(tmp_path / "mk.zip", {"L21_V001.csv": _map_keyframes_csv(26.44)})
    service = VideoPlaybackService(media, kmap)
    assert service.capture("L21_V001", 0.0756487).frame_idx == 2


def test_forward_and_reverse_with_offset(tmp_path: Path):
    media = _make_zip(tmp_path / "mi.zip", {"L21_V001.json": _media_info_json(length=2000)})
    kmap = _make_zip(tmp_path / "mk.zip", {"L21_V001.csv": _map_keyframes_csv(30.0)})
    service = VideoPlaybackService(media, kmap, offsets={"L21_V001": -172.0})

    meta = service.get_metadata("L21_V001")
    assert meta.playback_offset_seconds == -172.0

    # Player is at 180s; dataset timeline is 180 - (-172) = 352s in.
    result = service.capture("L21_V001", 180.0)
    assert result.source_time_seconds == pytest.approx(352.0)
    assert result.frame_idx == math.floor(352.0 * 30.0 + 1e-6)

    # Inverse: that frame index should map back to ~180s of player time.
    start = service.playback_start_seconds("L21_V001", result.frame_idx)
    assert start == pytest.approx(180.0, abs=1e-3)


def test_missing_video_raises_not_found(flat_media_info: Path, flat_map_keyframes: Path):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    with pytest.raises(VideoNotFoundError):
        service.get_metadata("L99_V999")


def test_missing_map_keyframes_raises_not_found(tmp_path: Path, flat_media_info: Path):
    empty_map = _make_zip(tmp_path / "empty-mk.zip", {"README.txt": "no csv here"})
    service = VideoPlaybackService(flat_media_info, empty_map)
    with pytest.raises(VideoNotFoundError):
        service.get_metadata("L21_V001")


def test_missing_watch_url_raises_metadata_error(tmp_path: Path, flat_map_keyframes: Path):
    media = _make_zip(
        tmp_path / "mi-no-url.zip",
        {"L21_V001.json": json.dumps({"title": "x", "length": 100})},
    )
    service = VideoPlaybackService(media, flat_map_keyframes)
    with pytest.raises(VideoMetadataError):
        service.get_metadata("L21_V001")


def test_missing_fps_never_falls_back_to_25(tmp_path: Path, flat_media_info: Path):
    bad_csv = "n,pts_time,frame_idx\n1,0.0,0\n"
    kmap = _make_zip(tmp_path / "mk-no-fps.zip", {"L21_V001.csv": bad_csv})
    service = VideoPlaybackService(flat_media_info, kmap)
    with pytest.raises(VideoMetadataError):
        service.get_metadata("L21_V001")


def test_blank_fps_values_raise_metadata_error(tmp_path: Path, flat_media_info: Path):
    bad_csv = "n,pts_time,fps,frame_idx\n1,0.0,,0\n2,3.0,0,90\n"
    kmap = _make_zip(tmp_path / "mk-blank-fps.zip", {"L21_V001.csv": bad_csv})
    service = VideoPlaybackService(flat_media_info, kmap)
    with pytest.raises(VideoMetadataError):
        service.get_metadata("L21_V001")


def test_negative_timestamp_rejected(flat_media_info: Path, flat_map_keyframes: Path):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    with pytest.raises(VideoRequestError):
        service.capture("L21_V001", -1.0)


def test_timestamp_beyond_duration_rejected(flat_media_info: Path, flat_map_keyframes: Path):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    with pytest.raises(VideoRequestError):
        service.capture("L21_V001", 5000.0)


def test_negative_frame_idx_rejected(flat_media_info: Path, flat_map_keyframes: Path):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    with pytest.raises(VideoRequestError):
        service.playback_start_seconds("L21_V001", -5)


def test_metadata_is_cached(flat_media_info: Path, flat_map_keyframes: Path, monkeypatch):
    service = VideoPlaybackService(flat_media_info, flat_map_keyframes)
    first = service.get_metadata("L21_V001")

    def _boom(*_args, **_kwargs):
        raise AssertionError("media-info should not be re-read after caching")

    monkeypatch.setattr(service, "_load_media_info", _boom)
    second = service.get_metadata("L21_V001")
    assert first is second


def test_default_map_keyframes_source_exists_for_capture():
    settings = Settings(_env_file=None)
    source = settings.get_map_keyframes_path()

    assert source.exists(), f"default map-keyframes source is missing: {source}"
    extracted_dir = source.parent / "map-keyframes"
    if extracted_dir.exists():
        assert source == extracted_dir
        assert any(source.glob("*.csv")), "map-keyframes directory contains no CSV data"
    else:
        assert source == extracted_dir.with_suffix(".zip")
        with zipfile.ZipFile(source) as archive:
            assert any(Path(name).suffix.lower() == ".csv" for name in archive.namelist())
