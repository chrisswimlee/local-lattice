"""First-run bootstrap: probe local backends and generate role registry JSON.

``local-lattice init`` uses this module to discover LM Studio or MLX model
inventories and write ``lmstudio_roles.json`` / ``mlx_roles.json`` with
sensible ``role:*`` mappings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import requests

from middle_layer.lmstudio_client import LMStudioClient, is_chat_capable_model_id

Backend = Literal["lmstudio", "mlx"]
InitBackend = Literal["auto", "lmstudio", "mlx"]

DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
ROLE_NAMES = ("fast", "coder", "reasoner", "vision", "default")

_EMBED_HINT = re.compile(
    r"(?ix)(?:^|[/_-])(?:embed|embedding|embeddings|nomic-embed|bge|e5)(?:[/_-]|$)"
)
_CODER_HINT = re.compile(r"(?ix)(?:coder|code-instruct|qwen.*coder|deepseek-coder)")
_VISION_HINT = re.compile(r"(?ix)(?:\bvl\b|vision|llava|gemma-4|pixtral)")
_FAST_HINT = re.compile(
    r"(?ix)(?:\b[3789]b\b|mini|small|tiny|phi-|granite-4\.1-8b|gpt-oss-20b|optiq)"
)
_REASONER_HINT = re.compile(
    r"(?ix)(?:"
    r"(?<![a-z])\b(?:12[0-9]|[7-9]\d|\d{3,})b\b"  # 70b+, 120–129b, 100b+ (not MoE a10b)
    r"|reason|thinking|r1|nemotron|hermes-4|opus|deckard"
    r")"
)


@dataclass(frozen=True)
class ProbeResult:
    backend: Backend | None
    lmstudio_url: str | None
    mlx_root: str | None
    model_ids: tuple[str, ...]
    loaded_ids: tuple[str, ...]
    ollama_detected: bool
    error: str | None


def discover_model_roots() -> list[Path]:
    """Return existing candidate directories for MLX weight scans."""
    candidates = [
        Path(os.environ.get("MLX_MODEL_ROOT", "")).expanduser(),
        Path("~/.lmstudio/models").expanduser(),
        Path("~/.cache/lm-studio/models").expanduser(),
        Path("~/.cache/mlx-models").expanduser(),
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not str(path) or not path.is_dir():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def scan_mlx_aliases(root: Path) -> list[str]:
    """Walk ``root`` the same way the MLX gateway registry scan does."""
    aliases: list[str] = []
    if not root.is_dir():
        return aliases
    try:
        entries = list(os.scandir(root))
    except OSError:
        return aliases
    for entry in entries:
        if not entry.is_dir():
            continue
        cfg = os.path.join(entry.path, "config.json")
        if os.path.exists(cfg):
            aliases.append(entry.name)
            continue
        try:
            sub_entries = list(os.scandir(entry.path))
        except OSError:
            continue
        for sub in sub_entries:
            if sub.is_dir() and os.path.exists(os.path.join(sub.path, "config.json")):
                aliases.append(f"{entry.name}/{sub.name}")
    return sorted(aliases, key=str.lower)


def probe_lmstudio(url: str = DEFAULT_LMSTUDIO_URL) -> ProbeResult:
    client = LMStudioClient(url)
    installed, err = client.get_model_ids(force_refresh=True)
    if err:
        return ProbeResult(None, None, None, (), (), False, err)
    loaded, _ = client.get_loaded_model_ids(force_refresh=True)
    chat_installed = tuple(m for m in installed if is_chat_capable_model_id(m))
    chat_loaded = tuple(m for m in loaded if is_chat_capable_model_id(m))
    if not chat_installed:
        return ProbeResult(
            None,
            None,
            None,
            (),
            (),
            False,
            "LM Studio responded but no chat-capable models were found.",
        )
    return ProbeResult(
        "lmstudio",
        client.base_url,
        None,
        chat_installed,
        chat_loaded,
        False,
        None,
    )


def probe_ollama(url: str = DEFAULT_OLLAMA_URL) -> bool:
    try:
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=2.0)
    except requests.RequestException:
        return False
    if resp.status_code != 200:
        return False
    data = resp.json()
    models = data.get("models") if isinstance(data, dict) else None
    return isinstance(models, list) and len(models) > 0


def infer_roles(model_id: str) -> set[str]:
    text = model_id.lower()
    if _EMBED_HINT.search(text):
        return set()
    roles: set[str] = set()
    if _CODER_HINT.search(text):
        roles.add("coder")
    if _VISION_HINT.search(text):
        roles.add("vision")
    if _FAST_HINT.search(text):
        roles.add("fast")
    if _REASONER_HINT.search(text):
        roles.add("reasoner")
    if not roles:
        roles.add("default")
    return roles


def classify_models(
    model_ids: list[str] | tuple[str, ...],
    *,
    loaded_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Map discovered ids into role buckets with loaded models listed first."""
    loaded = list(loaded_ids or [])
    loaded_set = set(loaded)
    chat_ids = [m for m in model_ids if is_chat_capable_model_id(m)]

    buckets: dict[str, list[str]] = {name: [] for name in ROLE_NAMES}
    for model_id in chat_ids:
        for role in infer_roles(model_id):
            if model_id not in buckets[role]:
                buckets[role].append(model_id)

    def _order(role_list: list[str]) -> list[str]:
        loaded_part = [m for m in loaded if m in role_list]
        rest = sorted((m for m in role_list if m not in loaded_set), key=str.lower)
        return loaded_part + rest

    out = {role: _order(buckets[role]) for role in ROLE_NAMES if buckets[role]}
    if "default" not in out and chat_ids:
        out["default"] = _order(chat_ids[:3])
    return out


def default_roles_path(backend: Backend, *, cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    return base / ("lmstudio_roles.json" if backend == "lmstudio" else "mlx_roles.json")


def build_roles_document(
    roles: dict[str, list[str]],
    *,
    backend: Backend,
    source: str,
    model_count: int,
    loaded_count: int,
) -> dict[str, object]:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    return {
        "_comment": (
            f"Auto-generated by local-lattice init on {stamp} for the {backend} gateway. "
            f"Source: {source}. Found {model_count} chat model(s)"
            f"{f', {loaded_count} loaded' if loaded_count else ''}. "
            "Re-run init after you change pinned/loaded models."
        ),
        **roles,
    }


def write_roles_file(
    path: Path,
    document: dict[str, object],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (pass --force to overwrite)")
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        print(text, end="")
        return
    path.write_text(text, encoding="utf-8")


def auto_probe(
    *,
    lmstudio_url: str = DEFAULT_LMSTUDIO_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> ProbeResult:
    lm = probe_lmstudio(lmstudio_url)
    if lm.backend is not None:
        return lm

    for root in discover_model_roots():
        aliases = scan_mlx_aliases(root)
        chat = [a for a in aliases if is_chat_capable_model_id(a)]
        if chat:
            return ProbeResult(
                "mlx",
                None,
                str(root),
                tuple(chat),
                (),
                probe_ollama(ollama_url),
                None,
            )

    ollama = probe_ollama(ollama_url)
    detail = lm.error or "No LM Studio server or MLX model directories were found."
    if ollama:
        detail += (
            " Ollama is running but is not a supported backend yet; "
            "use LM Studio or MLX weights."
        )
    return ProbeResult(None, None, None, (), (), ollama, detail)


def probe_backend(
    backend: InitBackend,
    *,
    lmstudio_url: str = DEFAULT_LMSTUDIO_URL,
    model_root: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> ProbeResult:
    if backend == "lmstudio":
        return probe_lmstudio(lmstudio_url)
    if backend == "mlx":
        roots = [Path(model_root).expanduser()] if model_root else discover_model_roots()
        for root in roots:
            aliases = scan_mlx_aliases(root)
            chat = [a for a in aliases if is_chat_capable_model_id(a)]
            if chat:
                return ProbeResult(
                    "mlx",
                    None,
                    str(root),
                    tuple(chat),
                    (),
                    probe_ollama(ollama_url),
                    None,
                )
        root_text = model_root or ", ".join(str(r) for r in roots) or "(none)"
        return ProbeResult(
            None,
            None,
            None,
            (),
            (),
            probe_ollama(ollama_url),
            f"No MLX models with config.json found under {root_text}.",
        )
    return auto_probe(lmstudio_url=lmstudio_url, ollama_url=ollama_url)


def print_next_steps(
    *,
    backend: Backend,
    roles_path: Path,
    probe: ProbeResult,
) -> None:
    print()
    if backend == "lmstudio":
        print(f"Detected LM Studio at {probe.lmstudio_url}")
        print(f"  chat models : {len(probe.model_ids)}")
        print(f"  loaded now  : {len(probe.loaded_ids)}")
        print(f"Wrote roles   : {roles_path}")
        print()
        print("Next steps:")
        print("  export MIDDLE_LAYER_API_KEY=$(uuidgen)")
        print("  ./scripts/start.sh --profile lmstudio")
        print("  ./scripts/demo.sh")
    else:
        print(f"Detected MLX models under {probe.mlx_root}")
        print(f"  chat models : {len(probe.model_ids)}")
        print(f"Wrote roles   : {roles_path}")
        print()
        print("Next steps:")
        print(f"  export MLX_MODEL_ROOT={probe.mlx_root!s}")
        print("  export MIDDLE_LAYER_API_KEY=$(uuidgen)")
        print("  ./scripts/start.sh --profile mlx")
        print("  BASE_URL=http://127.0.0.1:5001 ./scripts/demo.sh")
    if probe.ollama_detected:
        print()
        print(
            "Note: Ollama is running on this machine but Local Lattice does not "
            "proxy to it yet. LM Studio or MLX remain the supported backends."
        )


def run_init(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-lattice init",
        description="Probe local LLM backends and generate role registry JSON.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "lmstudio", "mlx"),
        default="auto",
        help="Which backend to probe (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Roles JSON path (default: ./lmstudio_roles.json or ./mlx_roles.json)",
    )
    parser.add_argument(
        "--lmstudio-url",
        default=os.environ.get("LM_STUDIO_URL", DEFAULT_LMSTUDIO_URL),
        help=f"LM Studio base URL (default: {DEFAULT_LMSTUDIO_URL})",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MLX_MODEL_ROOT"),
        help="MLX model root when --backend mlx (default: auto-discover)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing roles file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout instead of writing a file",
    )
    args = parser.parse_args(argv)

    probe = probe_backend(
        args.backend,
        lmstudio_url=args.lmstudio_url,
        model_root=args.model_root,
    )
    if probe.backend is None or probe.error:
        print(f"local-lattice init: {probe.error or 'nothing detected'}", file=sys.stderr)
        print(file=sys.stderr)
        print("Setup checklist:", file=sys.stderr)
        print("  1. Start LM Studio and load at least one chat model, or", file=sys.stderr)
        print("  2. Install MLX weights under ~/.lmstudio/models", file=sys.stderr)
        print("  3. Re-run: local-lattice init", file=sys.stderr)
        return 1

    roles = classify_models(probe.model_ids, loaded_ids=probe.loaded_ids)
    if not any(roles.values()):
        print("local-lattice init: no chat-capable models to classify.", file=sys.stderr)
        return 1

    backend: Backend = probe.backend
    out_path = args.output or default_roles_path(backend)
    if backend == "lmstudio":
        source = probe.lmstudio_url or args.lmstudio_url
    else:
        source = str(probe.mlx_root or args.model_root or "mlx-model-root")
    document = build_roles_document(
        roles,
        backend=backend,
        source=source,
        model_count=len(probe.model_ids),
        loaded_count=len(probe.loaded_ids),
    )

    try:
        write_roles_file(out_path, document, force=args.force, dry_run=args.dry_run)
    except FileExistsError as exc:
        print(f"local-lattice init: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        print_next_steps(backend=backend, roles_path=out_path.resolve(), probe=probe)
    return 0


__all__ = [
    "ProbeResult",
    "auto_probe",
    "classify_models",
    "default_roles_path",
    "discover_model_roots",
    "infer_roles",
    "probe_backend",
    "probe_lmstudio",
    "run_init",
    "scan_mlx_aliases",
    "write_roles_file",
]
