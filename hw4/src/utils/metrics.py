import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recovered, clean):
    assert recovered.shape == clean.shape
    recovered = np.clip(recovered.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)

    psnr_sum = 0.0
    ssim_sum = 0.0
    for i in range(recovered.shape[0]):
        psnr_sum += peak_signal_noise_ratio(clean[i], recovered[i], data_range=1)
        ssim_sum += structural_similarity(
            clean[i], recovered[i], data_range=1, channel_axis=-1,
        )

    n = recovered.shape[0]
    return psnr_sum / n, ssim_sum / n, n


class Timer:

    def __init__(self):
        self.acc = 0.0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0.0
        return ret

    def reset(self):
        self.acc = 0.0
