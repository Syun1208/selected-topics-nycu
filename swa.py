"""
Average weights across multiple checkpoints (SWA-style).

Usage:
    python swa.py work_dirs/r0/iter_15120.pth \
                  work_dirs/r0/iter_16128.pth \
                  work_dirs/r0/iter_17136.pth \
                  --out work_dirs/r0/swa_last3.pth

Then run inference:
    bash test.sh work_dirs/r0/swa_last3.pth
"""
import argparse
import torch
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpts', nargs='+', help='checkpoint paths to average')
    parser.add_argument('--out', required=True, help='output path')
    args = parser.parse_args()

    assert len(args.ckpts) >= 2, "Need at least 2 checkpoints to average"
    print(f"Averaging {len(args.ckpts)} checkpoints:")
    for c in args.ckpts:
        print(f"  - {c}")

    ckpts = [torch.load(c, map_location='cpu', weights_only=False)
             for c in args.ckpts]
    state_dicts = [c['state_dict'] for c in ckpts]

    keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if list(sd.keys()) != keys:
            raise ValueError(f"Checkpoint #{i} has different keys")

    n = len(state_dicts)
    avg = OrderedDict()
    for k in keys:
        t0 = state_dicts[0][k]
        if torch.is_tensor(t0) and t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for sd in state_dicts:
                acc += sd[k].to(torch.float64)
            avg[k] = (acc / n).to(t0.dtype)
        else:
            # int counters, etc.: take from the last (most recent) checkpoint
            avg[k] = state_dicts[-1][k]

    out_ckpt = {
        'state_dict': avg,
        'meta': ckpts[-1].get('meta', {}),
    }
    torch.save(out_ckpt, args.out)
    print(f"\nSaved averaged checkpoint -> {args.out}")
    print(f"Total params averaged: {sum(1 for k in keys if torch.is_tensor(state_dicts[0][k]) and state_dicts[0][k].is_floating_point())}")


if __name__ == '__main__':
    main()
