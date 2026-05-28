import math
import warnings
from collections import Counter
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1,
                 restarts=(0,), restart_weights=(1,), last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(self.restart_weights), \
            "restarts and their weights do not match."
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [g["initial_lr"] * weight for g in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [
            g["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for g in self.optimizer.param_groups
        ]


class LinearLR(_LRScheduler):

    def __init__(self, optimizer, total_iter, last_epoch=-1):
        self.total_iter = total_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        weight = 1 - self.last_epoch / self.total_iter
        return [weight * g["initial_lr"] for g in self.optimizer.param_groups]


def get_position_from_periods(iteration, cumulative_period):
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):

    def __init__(self, optimizer, periods, restart_weights=(1,),
                 eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert len(self.periods) == len(self.restart_weights), \
            "periods and restart_weights should have the same length."
        self.cumulative_period = [
            sum(self.periods[:i + 1]) for i in range(len(self.periods))
        ]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        period = self.periods[idx]
        return [
            self.eta_min + weight * 0.5 * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - nearest_restart) / period))
            for base_lr in self.base_lrs
        ]


class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                g["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, g in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                g["lr"] + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, g in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs)
                          / (self.max_epochs - self.warmup_epochs)))
            / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1)
                            / (self.max_epochs - self.warmup_epochs)))
            * (g["lr"] - self.eta_min) + self.eta_min
            for g in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch
                * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs)
                            / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
