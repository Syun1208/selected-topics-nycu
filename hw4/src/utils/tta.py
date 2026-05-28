import torch
import torch.nn.functional as F


def augment(x: torch.Tensor, op: int) -> torch.Tensor:
    if op >= 4:
        x = torch.flip(x, dims=[2])
    k = op % 4
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


def deaugment(x: torch.Tensor, op: int) -> torch.Tensor:
    k = op % 4
    if k:
        x = torch.rot90(x, -k, dims=[2, 3])
    if op >= 4:
        x = torch.flip(x, dims=[2])
    return x


def pad_input(x: torch.Tensor, img_multiple_of: int = 8):
    height, width = x.shape[2], x.shape[3]
    H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    x = F.pad(x, (0, padw, 0, padh), "reflect")
    return x, height, width


def tile_eval(model, x: torch.Tensor,
              tile: int = 128, tile_overlap: int = 32) -> torch.Tensor:
    b, c, h, w = x.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w).type_as(x)
    W_ = torch.zeros_like(E)

    for hi in h_idx_list:
        for wi in w_idx_list:
            in_patch = x[..., hi:hi + tile, wi:wi + tile]
            out_patch = model(in_patch)
            E[..., hi:hi + tile, wi:wi + tile].add_(out_patch)
            W_[..., hi:hi + tile, wi:wi + tile].add_(torch.ones_like(out_patch))
    return torch.clamp(E.div_(W_), 0, 1)


def predict_once(model, x: torch.Tensor,
                 tile: bool = False,
                 tile_size: int = 128,
                 tile_overlap: int = 32) -> torch.Tensor:
    if not tile:
        return model(x)
    x_pad, h, w = pad_input(x)
    out = tile_eval(model, x_pad, tile=tile_size, tile_overlap=tile_overlap)
    return out[:, :, :h, :w]


def predict(model, x: torch.Tensor,
            self_ensemble: bool = True,
            tile: bool = False,
            tile_size: int = 128,
            tile_overlap: int = 32) -> torch.Tensor:
    if not self_ensemble:
        return predict_once(
            model, x, tile=tile, tile_size=tile_size, tile_overlap=tile_overlap,
        )
    outs = []
    for op in range(8):
        ya = predict_once(
            model, augment(x, op),
            tile=tile, tile_size=tile_size, tile_overlap=tile_overlap,
        )
        outs.append(deaugment(ya, op))
    return torch.stack(outs, dim=0).mean(dim=0)
