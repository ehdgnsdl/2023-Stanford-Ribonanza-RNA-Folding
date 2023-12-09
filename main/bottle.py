from argparse import Namespace


import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import RNA_Model
from utils import sort_weight_decay_params
from apex.optimizers import FusedAdam as Adam


class RNA_Lightning(pl.LightningModule):
    def __init__(self, hp: Namespace):
        super(RNA_Lightning, self).__init__()
        self.save_hyperparameters(hp)
        self.hp = self.hparams
        self.model = RNA_Model(**vars(hp))

    def configure_optimizers(self):
        if self.hp.wt_decay:
            decay, no_decay = sort_weight_decay_params(self.model)
            opt_groups = [
                {"params": decay, "weight_decay": self.hp.wt_decay},
                {"params": no_decay, "weight_decay": 0},
            ]
            opt = Adam(opt_groups)
        else:
            opt = Adam(self.model.parameters(), weight_decay=0)
        return opt

    def on_train_start(self):
        self.n_steps = self.trainer.estimated_stepping_batches
        self.n_warmup_steps = self.n_steps * self.hp.lr_warmup
        self.hp_metric = 0.2
        self.logger.log_hyperparams(self.hp, {"hp/metric": self.hp_metric})

    def on_train_epoch_start(self):
        t = self.trainer
        self.logger.log_metrics({"hp/epoch": t.current_epoch}, t.global_step)

    def loss_L1(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l = F.l1_loss(x, y, reduction="none")
        return l[~torch.isnan(l)].mean()

    def loss_CE(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ls = self.hp.aux_smooth if self.training else 0
        x = x.transpose(1, 2)
        l = F.cross_entropy(x, y, reduction="none", label_smoothing=ls, ignore_index=0)
        return l[~torch.isnan(l)].mean()

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def fit_forward(self, batch: dict) -> torch.Tensor:
        prefix = "loss/T" if self.training else "loss/V"
        x = self(batch)
        x["react"] = x["react"] if self.training else x["react"].clip(0, 1)
        react = self.loss_L1(x["react"], batch["react"])
        loop = self.loss_CE(x["loop"], batch["loop"]) if self.hp.aux_loop else 0
        struct = self.loss_CE(x["struct"], batch["struct"]) if self.hp.aux_struct else 0
        log_d = {prefix: react} | ({f"{prefix}/loop": loop} if loop else {})
        log_d = log_d | ({f"{prefix}/struct": struct} if struct else {})
        self.log_dict(log_d, on_step=False, on_epoch=True)
        aux = (loop + struct) / max(bool(loop) + bool(struct), 1)
        return react + self.hp.aux_scale * aux

    def update_LR(self):
        lr_m = inv_sqrt_sched(self.trainer.global_step, self.n_warmup_steps)
        self.log("hp/lr", (lr := self.hp.lr * lr_m))
        for p in self.optimizers().optimizer.param_groups:
            p["lr"] = lr

    def training_step(self, batch, batch_idx):
        self.update_LR()
        return self.fit_forward(batch)

    def validation_step(self, batch, batch_idx):
        self.fit_forward(batch)

    def on_validation_end(self):
        l = self.trainer.logged_metrics["loss/V"].item()
        if l < self.hp_metric:
            self.hp_metric = l
            self.logger.log_metrics({"hp/metric": l}, self.trainer.global_step)

    def predict_step(self, batch, batch_idx):
        return self(batch)["react"].clip(0, 1)[batch["mask"]]


def inv_sqrt_sched(current_step: int, num_warmup_steps: int, timescale=None) -> float:
    timescale = num_warmup_steps if timescale is None else timescale
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / (((current_step + shift) / timescale) ** 0.5)
    return decay
