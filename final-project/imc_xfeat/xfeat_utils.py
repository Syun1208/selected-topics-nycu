"""Helpers to load the XFeat model that lives in the bundled ``accelerated_features`` repo."""

import os
import sys

# Make `accelerated_features` importable (it exposes the `modules` package).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_XFEAT_ROOT = os.path.join(_PROJECT_ROOT, "accelerated_features")
if _XFEAT_ROOT not in sys.path:
    sys.path.insert(0, _XFEAT_ROOT)

DEFAULT_WEIGHTS = os.path.join(_XFEAT_ROOT, "weights", "xfeat.pt")


def load_xfeat(weights=None, top_k=4096, detection_threshold=0.05):
    """Return an XFeat inference wrapper (``modules.xfeat.XFeat``).

    Used by the test/submission pipeline for feature extraction + matching.
    """
    from modules.xfeat import XFeat

    weights = weights or DEFAULT_WEIGHTS
    return XFeat(weights=weights, top_k=top_k, detection_threshold=detection_threshold)


def load_xfeat_model(weights=None, device="cpu"):
    """Return a raw ``XFeatModel`` (used for fine-tuning)."""
    import torch
    from modules.model import XFeatModel

    weights = weights or DEFAULT_WEIGHTS
    net = XFeatModel()
    if weights:
        net.load_state_dict(torch.load(weights, map_location=device))
    return net.to(device)
