"""Cloud manifest builder (scripts/cloud/build_asset_manifest.py)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "build_asset_manifest",
    os.path.join(BACKEND_ROOT, "scripts", "cloud", "build_asset_manifest.py"),
)
bam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bam)

from src.services.assets.manifest import parse_manifest  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestBuildManifestDoc:
    def test_explicit_hash_builds_valid_manifest(self) -> None:
        doc = bam.build_manifest_doc(
            "v1",
            [
                {"name": "faiss_index", "key": "beit3/f.index", "sha256": _sha(b"a"), "size": 1},
                {"name": "checkpoint", "key": "beit3/m.pth", "sha256": _sha(b"bb"), "size": 2},
            ],
            {"prefix": "kf"},
        )
        assert doc["version"] == "v1" and "generated_at" in doc
        # default containers per artifact name
        by = {a["name"]: a for a in doc["artifacts"]}
        assert by["faiss_index"]["container"] == "embeddings"
        assert by["checkpoint"]["container"] == "metadata"
        # keyframes defaults merged with the override
        assert doc["keyframes"]["prefix"] == "kf"
        assert doc["keyframes"]["layout"].endswith(".webp")
        # dogfood: the real parser accepts it
        manifest = parse_manifest(json.dumps(doc))
        assert {a.name for a in manifest.artifacts} == {"faiss_index", "checkpoint"}

    def test_hashes_local_files(self, tmp_path: Path) -> None:
        f = tmp_path / "global_ids.parquet"
        f.write_bytes(b"parquet-bytes-1234")
        doc = bam.build_manifest_doc(
            "v2", [{"name": "global_ids", "key": "g.parquet", "local": str(f)}], None
        )
        art = doc["artifacts"][0]
        assert art["sha256"] == _sha(b"parquet-bytes-1234")
        assert art["size"] == len(b"parquet-bytes-1234")

    def test_missing_local_file_errors(self) -> None:
        with pytest.raises(SystemExit):
            bam.build_manifest_doc("v3", [{"name": "x", "key": "x", "local": "/no/such/file"}], None)


class TestCliHelpers:
    def test_parse_artifact_flag_with_target(self) -> None:
        spec = bam._parse_artifact_flag("faiss_index=/data/f.index@embeddings/beit3/f.index")
        assert spec == {
            "name": "faiss_index", "local": "/data/f.index",
            "container": "embeddings", "key": "beit3/f.index",
        }

    def test_parse_artifact_flag_without_target_defaults_key_to_basename(self) -> None:
        spec = bam._parse_artifact_flag("tokenizer=/data/beit3.spm")
        assert spec["key"] == "beit3.spm" and "container" not in spec

    def test_parse_artifact_flag_rejects_bad_input(self) -> None:
        with pytest.raises(SystemExit):
            bam._parse_artifact_flag("no-equals-sign")

    def test_load_spec_dict_form(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(
            json.dumps(
                {
                    "version": "vX",
                    "artifacts": {
                        "faiss_index": {"local": "/data/f.index", "key": "beit3/f.index"},
                    },
                    "keyframes": {"container": "kf"},
                }
            ),
            encoding="utf-8",
        )
        version, artifacts, keyframes = bam._load_spec(spec_file)
        assert version == "vX"
        assert artifacts[0]["name"] == "faiss_index" and artifacts[0]["key"] == "beit3/f.index"
        assert keyframes == {"container": "kf"}


class TestMainWritesFile:
    def test_main_spec_to_out(self, tmp_path: Path) -> None:
        art = tmp_path / "f.index"
        art.write_bytes(b"IDX")
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(
            json.dumps(
                {"version": "v9", "artifacts": {"faiss_index": {"local": str(art), "key": "beit3/f.index"}}}
            ),
            encoding="utf-8",
        )
        out = tmp_path / "hcmai-assets.json"
        rc = bam.main(["--spec", str(spec_file), "--out", str(out)])
        assert rc == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["version"] == "v9"
        assert doc["artifacts"][0]["sha256"] == _sha(b"IDX")
