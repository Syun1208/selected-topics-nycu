import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import lightning.pytorch as pl

from src.models.promptir import build_promptir
from src.utils.losses import CharbonnierLoss, edge_loss, fft_l1_loss
from src.utils.schedulers import LinearWarmupCosineAnnealingLR
from src.utils.metrics import compute_psnr_ssim


class PromptIRLightningModule(pl.LightningModule):

    def __init__(
        self,
        model_cfg: dict = None,
        loss_cfg: dict = None,
        optimizer_cfg: dict = None,
        scheduler_cfg: dict = None,
        log_image_every: int = 6000,
    ):
        super().__init__()
        self.save_hyperparameters({
            "model_cfg": model_cfg or {},
            "loss_cfg": loss_cfg or {},
            "optimizer_cfg": optimizer_cfg or {},
            "scheduler_cfg": scheduler_cfg or {},
            "log_image_every": log_image_every,
        })

        self.net = build_promptir(self.hparams.model_cfg)

        loss_type = self.hparams.loss_cfg.get("type", "l1").lower()
        if loss_type == "charbonnier":
            eps = self.hparams.loss_cfg.get("charbonnier_eps", 1e-3)
            self.pixel_loss = CharbonnierLoss(eps=eps)
        else:
            self.pixel_loss = nn.L1Loss()

        self.w_pixel = float(self.hparams.loss_cfg.get("w_pixel", 1.0))
        self.w_edge = float(self.hparams.loss_cfg.get("w_edge", 0.1))
        self.w_fft = float(self.hparams.loss_cfg.get("w_fft", 0.0))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad, clean) = batch
        restored = self.net(degrad)

        bs = degrad.size(0)
        l_pix = self.pixel_loss(restored, clean)
        loss = self.w_pixel * l_pix
        log_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

        pixel_name = (
            "charbonnier_loss"
            if isinstance(self.pixel_loss, CharbonnierLoss)
            else "l1_loss"
        )
        self.log(pixel_name, l_pix, **log_kwargs)

        if self.w_edge > 0:
            l_edge = edge_loss(restored, clean)
            loss = loss + self.w_edge * l_edge
            self.log("edge_loss", l_edge, **log_kwargs)

        if self.w_fft > 0:
            l_fft = fft_l1_loss(restored, clean)
            loss = loss + self.w_fft * l_fft
            self.log("fft_loss", l_fft, **log_kwargs)

        self.log("train_loss", loss, **log_kwargs)

        psnr, ssim, _ = compute_psnr_ssim(restored, clean)
        self.log("psnr", psnr, **log_kwargs)
        self.log("ssim", ssim, **log_kwargs)

        if (
            self.logger is not None
            and self.hparams.log_image_every > 0
            and batch_idx % self.hparams.log_image_every == 0
        ):
            self._log_sample_grid(degrad, restored, clean)

        return loss

    def _log_sample_grid(self, degrad, restored, clean):
        try:
            from lightning.pytorch.loggers import WandbLogger
            import wandb
        except ImportError:
            return
        if not isinstance(self.logger, WandbLogger):
            return
        grid = torchvision.utils.make_grid(
            torch.cat([degrad[:4], restored[:4], clean[:4]], dim=0),
            nrow=4, normalize=True, scale_each=True,
        )
        self.logger.experiment.log({
            "Sample Input/Output/GT": wandb.Image(grid),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
        })

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer_cfg
        sch_cfg = self.hparams.scheduler_cfg

        lr = float(opt_cfg.get("lr", 2e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        optimizer = optim.AdamW(
            self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=int(sch_cfg.get("warmup_epochs", 15)),
            max_epochs=int(sch_cfg.get("max_epochs", 150)),
            eta_min=float(sch_cfg.get("eta_min", 1e-6)),
        )
        return [optimizer], [scheduler]
