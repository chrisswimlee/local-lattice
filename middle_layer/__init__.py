"""MiddleLayer: MLX-native OpenAI-compatible gateway.

This package is the future home of the gateway's importable Python API.
During the OSS-readiness migration the canonical implementation still
lives in the top-level ``middle_layerMLX`` (MLX backend) and
``middle_layer`` (LM Studio backend) modules at the repository root.

Subsequent passes will move that code under
``middle_layer.<subpackage>`` (see ``docs/_internal/RISK_REGISTER.md``
P3-01). Until then this package exposes:

- :func:`middle_layer.cli.main` — the console-script entry point
  ``middle-layer-mlx``. It forwards to the legacy module's ``main()``
  unchanged so existing behaviour is preserved bit-for-bit.

The package contains no other public API in this release. Do not import
private names; they will be reorganised without notice before 1.0.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
