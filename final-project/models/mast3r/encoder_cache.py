from __future__ import annotations

from typing import Any, Optional

import torch


class MASt3REncoderCache:
    """Per-scene cache of MASt3R encoder outputs.

    The MASt3R / DUSt3R encoder runs on each image independently — cross-
    attention only happens in the decoder. So the encoder output for image X is
    identical regardless of which other image X is paired with. This cache
    encodes each unique (image, crop) once per scene and reuses the result
    across every pair containing it, saving an entire ViT-L forward pass per
    repeat.

    Cache keys are caller-supplied so the matcher controls invalidation. Use
    `key=None` to bypass the cache (e.g. when an overlap-region cropper makes
    each pair's input genuinely different).
    """

    def __init__(self, max_entries: Optional[int] = None):
        self.max_entries = max_entries
        self._cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> dict[str, int]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._cache),
            "hit_rate_pct": int(100 * self._hits / total) if total else 0,
        }

    def release(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def _store(self, key: Any, feat: torch.Tensor, pos: torch.Tensor) -> None:
        self._cache[key] = (feat, pos)
        if self.max_entries is not None and len(self._cache) > self.max_entries:
            # FIFO eviction — scenes are small enough that this is rarely hit.
            oldest = next(iter(self._cache))
            del self._cache[oldest]

    @torch.inference_mode()
    def encode_pair(
        self,
        model,
        img1: torch.Tensor,
        shape1: torch.Tensor,
        img2: torch.Tensor,
        shape2: torch.Tensor,
        key1: Any = None,
        key2: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drop-in replacement for `model._encode_image_pairs` with caching.

        Returns (feat1, feat2, pos1, pos2) — same shapes & dtypes as the
        original method.
        """
        cached1 = self._cache.get(key1) if key1 is not None else None
        cached2 = self._cache.get(key2) if key2 is not None else None

        if cached1 is not None and cached2 is not None:
            self._hits += 2
            feat1, pos1 = cached1
            feat2, pos2 = cached2
            return feat1, feat2, pos1, pos2

        if cached1 is None and cached2 is None:
            # Both miss — use the batched path to mirror upstream behavior:
            # same shape → single forward with concatenated batch.
            self._misses += 2
            if img1.shape[-2:] == img2.shape[-2:]:
                out, pos, _ = model._encode_image(
                    torch.cat((img1, img2), dim=0),
                    torch.cat((shape1, shape2), dim=0),
                )
                feat1, feat2 = out.chunk(2, dim=0)
                pos1, pos2 = pos.chunk(2, dim=0)
            else:
                feat1, pos1, _ = model._encode_image(img1, shape1)
                feat2, pos2, _ = model._encode_image(img2, shape2)
            if key1 is not None:
                self._store(key1, feat1, pos1)
            if key2 is not None:
                self._store(key2, feat2, pos2)
            return feat1, feat2, pos1, pos2

        if cached1 is not None:
            self._hits += 1
            self._misses += 1
            feat1, pos1 = cached1
            feat2, pos2, _ = model._encode_image(img2, shape2)
            if key2 is not None:
                self._store(key2, feat2, pos2)
            return feat1, feat2, pos1, pos2

        # cached2 is not None
        self._hits += 1
        self._misses += 1
        feat2, pos2 = cached2  # type: ignore[misc]
        feat1, pos1, _ = model._encode_image(img1, shape1)
        if key1 is not None:
            self._store(key1, feat1, pos1)
        return feat1, feat2, pos1, pos2


def make_cache_key(path, true_shape: torch.Tensor) -> tuple[str, int, int]:
    """Build a stable cache key from path + view shape (H, W).

    Same image at the same resize/crop produces the same encoder output, so
    (path, H, W) is sufficient when no per-pair cropper is in effect.
    """
    h, w = int(true_shape[0, 0]), int(true_shape[0, 1])
    return (str(path), h, w)
