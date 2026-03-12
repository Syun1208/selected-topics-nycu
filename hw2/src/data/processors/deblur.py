import os
from typing import List, Optional

import cv2
import numpy as np

_NAFNET_CACHE: dict = {}

_DEFAULT_ARCH = dict(
    width=64,
    middle_blk_num=12,
    enc_blk_nums=[2, 2, 4, 8],
    dec_blk_nums=[2, 2, 2, 2],
)


def _import_nafnet_classes():
    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal

        return NAFNet, NAFNetLocal
    except ImportError as exc:
        raise ImportError(
            "NAFNet (basicsr) not found in sys.path.\n"
            "Register it with:\n"
            "  git clone https://github.com/megvii-research/NAFNet\n"
            "  echo /path/to/NAFNet > "
            '$(python -c "import site; print(site.getsitepackages()[0])")/nafnet.pth'
        ) from exc


def _parse_nafnet_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load NAFNet configs: pip install pyyaml") from exc

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    network_g = cfg.get("network_g", {})
    arch_type = network_g.get("type", "NAFNet")

    result = {
        "arch_type": arch_type,
        "width": network_g.get("width", _DEFAULT_ARCH["width"]),
        "middle_blk_num": network_g.get("middle_blk_num", _DEFAULT_ARCH["middle_blk_num"]),
        "enc_blk_nums": network_g.get("enc_blk_nums", _DEFAULT_ARCH["enc_blk_nums"]),
        "dec_blk_nums": network_g.get("dec_blk_nums", _DEFAULT_ARCH["dec_blk_nums"]),
        "checkpoint_path": None,
    }

    ckpt = cfg.get("path", {}).get("pretrain_network_g")
    if ckpt:
        ckpt = os.path.expandvars(os.path.expanduser(ckpt))
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(os.path.dirname(os.path.abspath(config_path)), ckpt)
        result["checkpoint_path"] = ckpt

    return result


def parse_gpu_ids(gpu_ids_str: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated GPU IDs string into a list of ints.

    Returns None if the string is empty or None (fall back to --nafnet-device).
    Examples: '0' → [0], '0,1,2' → [0, 1, 2], '' → None
    """
    if not gpu_ids_str:
        return None
    return [int(g.strip()) for g in gpu_ids_str.split(",") if g.strip()]


def _resolve_device(device: str, gpu_ids: Optional[List[int]]) -> str:
    """Return the primary torch device string given device and gpu_ids.

    If gpu_ids is provided, the primary device is cuda:<gpu_ids[0]>.
    Otherwise, device is used as-is.
    """
    if gpu_ids:
        return f"cuda:{gpu_ids[0]}"
    return device


def load_nafnet(
    checkpoint_path: str,
    device: str = "cpu",
    config_path: str = None,
    gpu_ids: Optional[List[int]] = None,
):
    """Load a NAFNet model, optionally reading arch from a YAML config and using multiple GPUs.

    Args:
        checkpoint_path: Path to a .pth checkpoint. If None, read from
                         ``path.pretrain_network_g`` in ``config_path``.
        device:          Torch device string ('cpu' or 'cuda'). Ignored when gpu_ids is set.
        config_path:     Optional path to a NAFNet YAML config file (e.g. NAFNet-width32.yml).
                         Architecture params are read from ``network_g`` section.
        gpu_ids:         Optional list of CUDA GPU IDs (e.g. [0] or [0, 1, 2]).
                         When set, overrides device; uses DataParallel for multiple GPUs.

    Returns:
        Loaded, eval()-mode model on the requested device (DataParallel if multiple gpu_ids).
    """
    import torch

    primary_device = _resolve_device(device, gpu_ids)

    parsed = None
    if config_path is not None:
        parsed = _parse_nafnet_config(config_path)
        if checkpoint_path is None:
            checkpoint_path = parsed.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                f"No checkpoint path provided and config '{config_path}' has no "
                "path.pretrain_network_g. Pass --deblurry-checkpoint-path explicitly."
            )

    cache_key = (
        os.path.abspath(checkpoint_path),
        primary_device,
        config_path,
        tuple(gpu_ids or []),
    )
    if cache_key in _NAFNET_CACHE:
        return _NAFNET_CACHE[cache_key]

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"NAFNet checkpoint not found: {checkpoint_path}\n"
            "Download GoPro (deblur)  → https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC\n"
            "Download SIDD  (denoise) → https://drive.google.com/file/d/14Fht4x2Ft4HEDMoBT4SRyiqgQ73YRa6I"
        )

    NAFNet, NAFNetLocal = _import_nafnet_classes()

    if parsed is not None:
        arch_type = parsed["arch_type"]
        arch_kwargs = dict(
            img_channel=3,
            width=parsed["width"],
            middle_blk_num=parsed["middle_blk_num"],
            enc_blk_nums=parsed["enc_blk_nums"],
            dec_blk_nums=parsed["dec_blk_nums"],
        )
        ModelClass = NAFNetLocal if arch_type == "NAFNetLocal" else NAFNet
    else:
        ModelClass = NAFNet
        arch_kwargs = dict(img_channel=3, **_DEFAULT_ARCH)

    model = ModelClass(**arch_kwargs)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params", state), strict=True)
    model.eval()
    model = model.to(primary_device)

    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    _NAFNET_CACHE[cache_key] = model
    return model


def _run_nafnet(
    image: np.ndarray,
    checkpoint_path: str,
    device: str = "cpu",
    config_path: str = None,
    gpu_ids: Optional[List[int]] = None,
) -> np.ndarray:
    import torch

    primary_device = _resolve_device(device, gpu_ids)
    model = load_nafnet(checkpoint_path, device, config_path=config_path, gpu_ids=gpu_ids)
    h, w = image.shape[:2]

    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    img_pad = (
        cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if (pad_h or pad_w)
        else image
    )

    img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(primary_device)

    with torch.no_grad():
        out = model(inp)

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr[:h, :w]


def deblur_nafnet(
    image: np.ndarray,
    checkpoint_path: str,
    device: str = "cpu",
    config_path: str = None,
    gpu_ids: Optional[List[int]] = None,
) -> np.ndarray:
    return _run_nafnet(image, checkpoint_path, device, config_path=config_path, gpu_ids=gpu_ids)
