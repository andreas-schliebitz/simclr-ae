import lightning as L
import torch.nn.functional as F

from torch import nn, Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from simclr_ae.models.logreg.utils import create_metrics


class LogisticRegressionModule(L.LightningModule):
    def __init__(
        self,
        epochs: int,
        latent_dim: int,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = nn.Linear(latent_dim, num_classes)

        # Macro metrics
        macro_metrics = create_metrics(
            num_classes=num_classes, task="multiclass", average="macro"
        )

        self.train_macro_metrics = macro_metrics.clone(prefix="train_macro_")
        self.val_macro_metrics = macro_metrics.clone(prefix="val_macro_")
        self.test_macro_metrics = macro_metrics.clone(prefix="test_macro_")

        # Micro metrics
        micro_metrics = create_metrics(
            num_classes=num_classes, task="multiclass", average="micro"
        )

        self.train_micro_metrics = micro_metrics.clone(prefix="train_micro_")
        self.val_micro_metrics = micro_metrics.clone(prefix="val_micro_")
        self.test_micro_metrics = micro_metrics.clone(prefix="test_micro_")

    def training_step(self, batch: tuple[Tensor, Tensor]) -> float:
        inputs, targets = batch
        logits = self.model(inputs)

        train_loss = F.cross_entropy(logits, targets)
        self.log("train_loss", train_loss, prog_bar=True, sync_dist=True)

        class_probabilities = F.softmax(logits, dim=1)

        macro_metrics = self.train_macro_metrics(class_probabilities, targets)
        self.log_dict(macro_metrics, sync_dist=True)

        micro_metrics = self.train_micro_metrics(class_probabilities, targets)
        self.log_dict(micro_metrics, sync_dist=True)

        return train_loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> float:
        inputs, targets = batch
        logits = self.model(inputs)

        val_loss = F.cross_entropy(logits, targets)
        self.log("val_loss", val_loss, sync_dist=True)

        class_probabilities = F.softmax(logits, dim=1)

        self.val_macro_metrics.update(class_probabilities, targets)
        self.val_micro_metrics.update(class_probabilities, targets)

        return val_loss

    def on_validation_epoch_end(self) -> None:
        macro_metrics = self.val_macro_metrics.compute()
        self.log_dict(macro_metrics, sync_dist=True)
        self.val_macro_metrics.reset()

        micro_metrics = self.val_micro_metrics.compute()
        self.log_dict(micro_metrics, sync_dist=True)
        self.val_micro_metrics.reset()

    def test_step(self, batch: tuple[Tensor, Tensor]) -> None:
        inputs, targets = batch
        logits = self.model(inputs)

        test_loss = F.cross_entropy(logits, targets)
        self.log("test_loss", test_loss, sync_dist=True)

        class_probabilities = F.softmax(logits, dim=1)

        self.test_macro_metrics.update(class_probabilities, targets)
        self.test_micro_metrics.update(class_probabilities, targets)

    def on_test_epoch_end(self) -> None:
        macro_metrics = self.test_macro_metrics.compute()
        self.log_dict(macro_metrics, sync_dist=True)
        self.test_macro_metrics.reset()

        micro_metrics = self.test_micro_metrics.compute()
        self.log_dict(micro_metrics, sync_dist=True)
        self.test_micro_metrics.reset()

    def configure_optimizers(self) -> dict:
        optimizer = SGD(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "train_loss"},
        }
