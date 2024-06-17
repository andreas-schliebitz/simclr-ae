import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as V

from torch import Tensor
from pathlib import Path
from PIL.Image import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.loggers import MLFlowLogger, CSVLogger


class AEModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 0,
        log_n_samples: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.log_n_samples = log_n_samples
        self.log_sample_count = 0

        self.test_metrics = []

    def compute_loss(self, r: Tensor, x: Tensor) -> float:
        batch_size = x.size(0)
        r = r.view(batch_size, -1)
        x = x.view(batch_size, -1)
        return F.mse_loss(r, x)

    def forward(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, labels = batch
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> float:
        x, labels = batch
        r = self.model(x)
        train_loss = self.compute_loss(r, x)
        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> float:
        x, labels = batch
        r = self.model(x)
        val_loss = self.compute_loss(r, x)
        self.log("val_loss", val_loss, sync_dist=True)
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.test_metrics = []

    def test_step(self, batch: tuple[Tensor, Tensor]) -> None:
        x, labels = batch
        r = self.model(x)
        test_mse = self.compute_loss(r, x)
        self.test_metrics.append(test_mse)

    def on_test_epoch_end(self) -> None:
        test_mse = torch.stack(self.test_metrics, dim=0).mean(dim=0)
        self.log("test_mse", test_mse, sync_dist=True)

    def predict_step(self, batch: tuple[Tensor, Tensor]) -> list[tuple[Image, Image]]:
        x, labels = batch
        r = self.model(x)
        return [
            (V.to_pil_image(x_img), V.to_pil_image(r_img)) for x_img, r_img in zip(x, r)
        ]

    def on_predict_batch_end(
        self,
        outputs: list[tuple[Image, Image]],
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for idx, (x_img, r_img) in enumerate(outputs):
            if self.log_sample_count >= self.log_n_samples - 1:
                break
            self._visualize_prediction(x_img, r_img, batch_idx=batch_idx, idx=idx)
            self.log_sample_count += 1

    # See: https://arxiv.org/pdf/2206.08309.pdf, page 25
    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }

    def _visualize_prediction(
        self,
        x_img: Image,
        r_img: Image,
        batch_idx: int,
        idx: int,
    ) -> None:
        batch_num = str(batch_idx).zfill(3)
        idx_num = str(idx).zfill(5)
        for img, postfix in ((x_img, "orig"), (r_img, "pred")):
            img_filename = f"b{batch_num}-s{idx_num}-{postfix}.png"
            for logger in self.loggers:
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_image(
                        image=img,
                        artifact_file=Path("predictions", img_filename),
                        run_id=logger.run_id,
                    ),
                elif isinstance(logger, CSVLogger):
                    pred_dir = Path(logger.log_dir, "predictions")
                    if not pred_dir.exists():
                        pred_dir.mkdir()
                    img.save(Path(pred_dir, img_filename))
