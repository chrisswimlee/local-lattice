"""Tests for middle_layer.bootstrap (local-lattice init)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import middle_layer.bootstrap as bootstrap


def test_infer_roles_skips_embeddings_and_tags_coder() -> None:
    assert bootstrap.infer_roles("text-embedding-nomic-embed-text-v1.5") == set()
    assert "coder" in bootstrap.infer_roles("qwen/qwen3-coder-next")


def test_classify_models_puts_loaded_first() -> None:
    roles = bootstrap.classify_models(
        ["granite-4.1-8b", "qwen/qwen3-coder-next", "qwen3.5-122b-a10b"],
        loaded_ids=["qwen3.5-122b-a10b"],
    )
    assert roles["coder"][0] == "qwen/qwen3-coder-next"
    assert roles["reasoner"][0] == "qwen3.5-122b-a10b"
    assert roles["fast"][0] == "granite-4.1-8b"


def test_scan_mlx_aliases_finds_publisher_layout(tmp_path: Path) -> None:
    pub = tmp_path / "mlx-community"
    model = pub / "Demo-7B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert bootstrap.scan_mlx_aliases(tmp_path) == ["mlx-community/Demo-7B"]


def test_write_roles_file_dry_run_and_force(tmp_path: Path, capsys) -> None:
    path = tmp_path / "lmstudio_roles.json"
    doc = {"fast": ["a"], "_comment": "test"}
    bootstrap.write_roles_file(path, doc, dry_run=True)
    assert not path.exists()
    assert '"fast"' in capsys.readouterr().out

    bootstrap.write_roles_file(path, doc, force=False)
    assert path.exists()
    try:
        bootstrap.write_roles_file(path, doc, force=False)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    bootstrap.write_roles_file(path, doc, force=True)
    assert json.loads(path.read_text())["fast"] == ["a"]


def test_probe_lmstudio_uses_client() -> None:
    client = MagicMock()
    client.base_url = "http://127.0.0.1:1234"
    client.get_model_ids.return_value = (["granite-4.1-8b", "text-embedding-foo"], None)
    client.get_loaded_model_ids.return_value = (["granite-4.1-8b"], None)
    with patch.object(bootstrap, "LMStudioClient", return_value=client):
        result = bootstrap.probe_lmstudio()
    assert result.backend == "lmstudio"
    assert result.model_ids == ("granite-4.1-8b",)
    assert result.loaded_ids == ("granite-4.1-8b",)


def test_run_init_writes_lmstudio_roles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    probe = bootstrap.ProbeResult(
        "lmstudio",
        "http://127.0.0.1:1234",
        None,
        ("granite-4.1-8b", "qwen/qwen3-coder-next"),
        ("granite-4.1-8b",),
        False,
        None,
    )
    with patch.object(bootstrap, "probe_backend", return_value=probe):
        code = bootstrap.run_init(["--backend", "lmstudio"])
    assert code == 0
    out = tmp_path / "lmstudio_roles.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "fast" in data
    assert data["fast"][0] == "granite-4.1-8b"


def test_run_init_failure_when_nothing_detected(capsys) -> None:
    probe = bootstrap.ProbeResult(None, None, None, (), (), False, "nope")
    with patch.object(bootstrap, "probe_backend", return_value=probe):
        code = bootstrap.run_init([])
    assert code == 1
    assert "nope" in capsys.readouterr().err
