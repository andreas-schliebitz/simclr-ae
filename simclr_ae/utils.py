import os
import yaml
import torch
import torchvision
import torch.nn as nn
import lightning as L

from typing import Any
from torch import Tensor
from pathlib import Path
from lightning.pytorch import seed_everything
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import Logger, CSVLogger, MLFlowLogger

RANDOM_SEED = 42


def seed_rng() -> None:
    seed_everything(seed=RANDOM_SEED, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def calc_split_sizes(
    dataset_size: int, train_perc: float, val_perc: float, test_perc: int
) -> tuple[int, int, int]:
    train_size = round(train_perc * dataset_size)
    val_size = round(train_size * val_perc)
    test_size = dataset_size - train_size
    train_size -= val_size

    assert train_size + val_size + test_size == dataset_size

    return train_size, val_size, test_size


def calc_offset_and_length(
    split: str, train_size: int, val_size: int, test_size: int
) -> tuple[int, int]:
    # [train | val | test]
    match split:
        case "train":
            offset = 0
            length = train_size
        case "val":
            offset = train_size
            length = val_size
        case "test":
            offset = train_size + val_size
            length = test_size
        case _:
            raise ValueError(f"Unknown split name: {split}")
    return offset, length


def print_dataloader_summary(
    dataloader: DataLoader, split: str, num_samples: int = 1
) -> None:
    print(f"{split.upper()} Dataloader Summary:")
    print(f" - Number of batches: {len(dataloader)}")
    print(f" - Number of samples: {len(dataloader.dataset)}")
    print(f" - Batch size: {dataloader.batch_size}")

    try:
        print(f"\nSample data from the first {num_samples} batches:")
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            for i, batch_component in enumerate(batch):
                print(f" - Component [{i}] shape: {batch_component.shape}")
            if batch_idx == num_samples:
                break
    except:
        print(f"Error extracting sample data from '{split.upper()}' dataloader.")


def print_dataset_summary(dataset: Dataset, name: str, num_samples: int = 1) -> None:
    print(f"{name.upper()} Dataset Summary:")
    print(f" - Number of samples: {len(dataset)}")

    try:
        print("\nSample data:")
        for sample_idx, sample in enumerate(dataset, start=1):
            print(f"Sample {sample_idx}:")
            for i, sample_component in enumerate(sample):
                print(
                    f" - Component [{i}]: {sample_component.shape}, {sample_component.dtype}"
                )
            if sample_idx == num_samples:
                break
    except:
        print(f"Error extracting sample data from dataset '{name}'.")


def get_backbone_model(backbone: str, weights: str | None = None) -> nn.Module:
    return getattr(torchvision.models, backbone)(weights=weights)


def create_trainer_callbacks(patience: int, save_top_k: int) -> dict[str, Callback]:
    trainer_callbacks = {}
    if patience > 0:
        trainer_callbacks["EarlyStopping"] = EarlyStopping(
            patience=patience, monitor="val_loss"
        )
    if save_top_k > 0:
        trainer_callbacks["ModelCheckpoint"] = ModelCheckpoint(
            save_top_k=save_top_k,
            auto_insert_metric_name=True,
            save_last=patience <= 0,
            monitor="val_loss",
        )
    return trainer_callbacks


def create_loggers(
    log_dir: str,
    experiment_name: str,
    run_name: str,
    use_mlflow: bool = False,
) -> dict[str, Logger]:
    loggers = {
        "CSVLogger": CSVLogger(
            save_dir=Path(log_dir, "csv", experiment_name),
            name=run_name,
        )
    }

    if use_mlflow:
        loggers["MLFlowLogger"] = MLFlowLogger(
            save_dir=str(Path(log_dir, "mlruns", experiment_name)),
            experiment_name=experiment_name,
            run_name=run_name,
            log_model=True,
        )

    return loggers


def print_tensor_stats(prefix: str, x: Tensor) -> None:
    print(f"=== Tensor - {prefix} ===")
    print(" - Mean:", x.to(torch.float).mean().item())
    print(" - Stdev:", x.to(torch.float).std().item())
    print(" - Min:", x.to(torch.float).min().item())
    print(" - Max:", x.to(torch.float).max().item())
    print(" - Shape:", tuple(x.shape))
    print(" - Dtype:", x.dtype)


def is_iterable(obj: Any) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def load_yaml(filepath: str | Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.Loader)


def get_parameter_stats(module: L.LightningModule, prefix: str = "") -> dict:
    return {
        f"{prefix}trainable_params": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
        f"{prefix}non_trainable_params": sum(
            p.numel() for p in module.parameters() if not p.requires_grad
        ),
    }
