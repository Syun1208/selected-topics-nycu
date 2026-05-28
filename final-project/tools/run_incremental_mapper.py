"""Subprocess wrapper for ``pycolmap.incremental_mapping``.

Why: pycolmap/pyceres registers a progress callback that calls back into Python
during Bundle Adjustment. Inside Ceres' worker code path the callback ends up
calling ``PyErr_Fetch`` from a non-Python thread (via ``pthread_once``-style
init in pyceres), which segfaults regardless of ``mapper_options.num_threads``.

Running ``incremental_mapping`` in a fresh subprocess isolates the crash from
the main pipeline: even if Ceres segfaults, the parent only sees a non-zero
return code and can fall back gracefully.

Usage::

    python tools/run_incremental_mapper.py \
        --database-path PATH \
        --image-path PATH \
        --output-path PATH \
        --options-json '{"num_threads": 1, "min_model_size": 3, ...}'

Reconstructions are written by pycolmap into ``output_path/0/``,
``output_path/1/``, ... so the parent can reload them with
``pycolmap.Reconstruction(output_path / "0")``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pycolmap


def _set_dotted(obj, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def apply_options(options: "pycolmap.IncrementalPipelineOptions", flat: dict) -> None:
    for k, v in flat.items():
        _set_dotted(options, k, v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--options-json",
        default="{}",
        help="Flat JSON object of pycolmap IncrementalPipelineOptions to set "
        "(supports dotted keys like 'mapper.filter_max_reproj_error').",
    )
    args = parser.parse_args()

    options = pycolmap.IncrementalPipelineOptions()
    try:
        flat = json.loads(args.options_json)
    except json.JSONDecodeError as e:
        print(f"[run_incremental_mapper] bad --options-json: {e}", file=sys.stderr)
        return 2

    if not isinstance(flat, dict):
        print(
            "[run_incremental_mapper] --options-json must be a JSON object",
            file=sys.stderr,
        )
        return 2

    try:
        apply_options(options, flat)
    except AttributeError as e:
        print(
            f"[run_incremental_mapper] unknown option in --options-json: {e}",
            file=sys.stderr,
        )
        return 2

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    pycolmap.incremental_mapping(
        database_path=args.database_path,
        image_path=args.image_path,
        output_path=args.output_path,
        options=options,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
