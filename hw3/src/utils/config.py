import os
from pathlib import Path
from typing import Any, Optional

import yaml


class Config:

    def __init__(self, data: dict, root: Optional[Path] = None):
        self._data = data
        self._root = root or Path.cwd()
        self._resolve_env(self._data)


    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data, root=Path(path).parent)


    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        if isinstance(val, dict):
            return Config(val, self._root)
        return val

    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        if isinstance(val, dict):
            return Config(val, self._root)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Config(val, self._root)
        return val

    def to_dict(self) -> dict:
        return self._data


    def _resolve_env(self, node: Any) -> Any:
        if isinstance(node, dict):
            for k, v in node.items():
                node[k] = self._resolve_env(v)
        elif isinstance(node, list):
            return [self._resolve_env(i) for i in node]
        elif isinstance(node, str) and node.startswith("${") and node.endswith("}"):
            var = node[2:-1]
            return os.environ.get(var, node)
        return node

    def __repr__(self) -> str:
        return f"Config({self._data})"


def build_output_paths(cfg: dict) -> dict:
    model_name = cfg["model"]["name"]
    backbone = cfg["model"]["backbone"]
    exp_name = cfg.get("experiment", {}).get("name", "v1")
    base = Path(cfg["paths"]["base_dir"])
    return {
        "checkpoint_dir": str(base / "checkpoints" / model_name / backbone / exp_name),
        "chart_dir": str(base / "charts" / model_name / backbone / exp_name),
    }
