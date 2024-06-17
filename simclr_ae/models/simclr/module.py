import torch
import lightning as L
import torch.nn as nn
import torchvision.transforms.functional as V

from pathlib import Path
from torch import Tensor
from torch.optim import SGD
from collections import defaultdict
from pythae.models.base.base_utils import ModelOutput
from torch.optim.lr_scheduler import CosineAnnealingLR
from simclr_ae.models.simclr.loss import NTXEntCriterion
from lightning.pytorch.loggers import MLFlowLogger, CSVLogger


class SimCLRModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 10e-6,
        temperature: float = 0.5,
        log_n_samples: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.model = model
        self.epochs = epochs
        self.loss = NTXEntCriterion(temperature)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.log_n_samples = log_n_samples
        self.log_sample_count = defaultdict(int)

    def compute_loss(
        self, z_i: Tensor | ModelOutput, z_j: Tensor | ModelOutput
    ) -> Tensor:
        return self.loss(torch.concat((z_i, z_j)))

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor | None], batch_idx: int
    ) -> float:
        x, x_i, x_j, labels = batch
        z_i, z_j = self.model(x_i, x_j)
        train_loss = self.compute_loss(z_i, z_j)
        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss

    def on_train_batch_end(
        self,
        outputs: list[float],
        batch: tuple[Tensor, Tensor, Tensor, Tensor | None],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        split = "train"
        x, x_i, x_j, labels = batch
        for idx, (x_elem, x_i_elem, x_j_elem, label) in enumerate(
            zip(x, x_i, x_j, labels)
        ):
            if self.log_sample_count[split] >= self.log_n_samples - 1:
                return
            self._visualize_sample(
                x_elem,
                x_i_elem,
                x_j_elem,
                batch_idx=batch_idx,
                idx=idx,
                split=split,
            )

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor | None], batch_idx: int
    ) -> float:
        x, x_i, x_j, labels = batch
        z_i, z_j = self.model(x_i, x_j)
        val_loss = self.compute_loss(z_i, z_j)
        self.log("val_loss", val_loss, sync_dist=True)
        return val_loss

    def on_validation_batch_end(
        self,
        outputs: list[float],
        batch: tuple[Tensor, Tensor, Tensor, Tensor | None],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        split = "val"
        x, x_i, x_j, labels = batch
        for idx, (x_elem, x_i_elem, x_j_elem, label) in enumerate(
            zip(x, x_i, x_j, labels)
        ):
            if self.log_sample_count[split] >= self.log_n_samples - 1:
                return
            self._visualize_sample(
                x_elem,
                x_i_elem,
                x_j_elem,
                batch_idx=batch_idx,
                idx=idx,
                split=split,
            )

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor | None]:
        x_img, x, labels = batch
        features = self.model.f(x)
        return features, labels

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.learning_rate / 50
        )

        return [optimizer], [lr_scheduler]

    def _visualize_sample(
        self,
        x: Tensor,
        x_i: Tensor,
        x_j: Tensor,
        batch_idx: int,
        idx: int,
        split: str,
    ) -> None:
        batch_num = str(batch_idx).zfill(3)
        idx_num = str(idx).zfill(5)
        for img, kind in ((x, "orig"), (x_i, "t1"), (x_j, "t2")):
            img = V.to_pil_image(img)
            img_filename = f"{split}-b{batch_num}-s{idx_num}-{kind}.png"
            for logger in self.loggers:
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_image(
                        image=img,
                        artifact_file=Path("transforms", img_filename),
                        run_id=logger.run_id,
                    )
                elif isinstance(logger, CSVLogger):
                    transforms_dir = Path(logger.log_dir, "transforms")
                    if not transforms_dir.exists():
                        transforms_dir.mkdir()
                    img.save(Path(transforms_dir, img_filename))
        self.log_sample_count[split] += 1
