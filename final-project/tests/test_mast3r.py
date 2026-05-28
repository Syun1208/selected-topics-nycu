from __future__ import annotations

import numpy as np
import torch
from dust3r.utils.geometry import xy_grid


def test_xy_grid():
    H, W = (100, 100)

    conf = np.random.rand(H, W) > 0.5
    print(conf)
    pts = xy_grid(W, H).reshape(-1, 2)[conf.flatten()]
    print(pts)


def test_grid_sample():
    H, W = (100, 100)

    kpts = torch.rand((30, 2)).cuda()
    kpts[:, 0] = kpts[:, 0] * W
    kpts[:, 1] = kpts[:, 1] * H

    descs = torch.rand((H, W, 24)).to(kpts.device)
    descs = descs.permute(2, 0, 1).unsqueeze(0)
    print(descs.shape)

    grid = (
        torch.stack(
            [
                2.0 * kpts[:, 0] / (W - 1) - 1,  # Normalize x
                2.0 * kpts[:, 1] / (H - 1) - 1,  # Normalize y
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(2)
        .cuda()
    )
    print(grid)
    descs = torch.nn.functional.grid_sample(
        descs, grid, align_corners=True, mode="bilinear"
    )  # Shape: (1, C, N, 1)
    print(descs.shape)
