import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


def edge_loss(pred, target):

    def sobel(x):
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32, device=x.device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        gx = F.conv2d(x, sobel_x, padding=1, groups=1)
        gy = F.conv2d(x, sobel_y, padding=1, groups=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    pred_gray = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
    target_gray = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]
    return F.l1_loss(sobel(pred_gray), sobel(target_gray))


def fft_l1_loss(pred, target):
    pred_f = torch.fft.rfft2(pred, norm="ortho")
    targ_f = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_f), torch.abs(targ_f))


class GANLoss(nn.Module):

    def __init__(self, use_lsgan: bool = True,
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.numel() != input.numel():
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            return self.real_label_var
        if self.fake_label_var is None or self.fake_label_var.numel() != input.numel():
            self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
        return self.fake_label_var

    def __call__(self, input, target_is_real):
        return self.loss(input, self.get_target_tensor(input, target_is_real))
