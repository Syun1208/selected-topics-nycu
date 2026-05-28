from typing import Optional

import torch
import lightning.pytorch as pl


class ModelEma:

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().cpu()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.items():
            src = msd[k].detach().to(v.device, non_blocking=True)
            if v.is_floating_point():
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)


class EMACallback(pl.Callback):

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema: Optional[ModelEma] = None
        self._pending_state = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = ModelEma(pl_module.net, decay=self.decay)
        if self._pending_state is not None:
            for k, v in self._pending_state.items():
                if k in self.ema.shadow:
                    self.ema.shadow[k].copy_(v.to(self.ema.shadow[k].device))
            self._pending_state = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(pl_module.net)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = {
                k: v.detach().cpu() for k, v in self.ema.shadow.items()
            }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            self._pending_state = checkpoint["ema_state_dict"]
