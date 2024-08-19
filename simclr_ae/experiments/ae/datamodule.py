import os
import numpy as np
import torch.nn as nn
import lightning as L

from torchvision import transforms
from lightning.pytorch.loggers import Logger
from torch.utils.data import Dataset, DataLoader
from simclr_ae.models.ae.dataset import AEDataset
from simclr_ae.utils import print_dataloader_summary, is_iterable
from simclr_ae.dataset import (
    get_vision_dataset,
    split_vision_dataset,
    calculate_dataset_stats,
)


class AEDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str,
        train_perc: float,
        val_perc: float,
        test_perc: float,
        img_size: int,
        batch_size: int = 32,
        standardize: bool = False,
        num_workers: int = -1,
        logger: Logger | list[Logger] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.train_perc = train_perc
        self.val_perc = val_perc
        self.test_perc = test_perc
        self.img_size = img_size
        self.batch_size = batch_size
        self.standardize = standardize
        self.num_workers = os.cpu_count() if num_workers <= 0 else num_workers
        self.logger = (
            [logger] if logger is not None and not is_iterable(logger) else logger
        )

        self.dataset = None
        self.num_classes = None
        self.train_split = None
        self.val_split = None
        self.test_split = None
        self.predict_split = None

    def log_hyperparams(self, params: dict) -> None:
        if self.logger is not None:
            for logger in self.logger:
                logger.log_hyperparams(params)

    def default_transforms(
        self,
        mean: np.ndarray[float] | None = None,
        std: np.ndarray[float] | None = None,
        compose: bool = False,
    ) -> list[nn.Module] | transforms.Compose:
        transformations = [
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]
        if self.standardize and mean is not None and std is not None:
            assert np.all((mean >= 0) & (mean <= 1)) and np.all(
                (std >= 0) & (std <= 1)
            ), "Mean and std values have to be between [0,1] as 'ToTensor()' automatically scales to [0,1]."
            transformations.append(transforms.Normalize(mean=mean, std=std))
            print(f"Normalizing train data using mean={mean} and std={std} values.")
        if compose:
            return transforms.Compose(transformations)
        return transformations

    def _get_dataset_split(
        self, split: str
    ) -> tuple[Dataset, np.ndarray[float], np.ndarray[float]]:
        dataset, num_classes, has_splits = get_vision_dataset(
            split=split,
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
        )
        self.dataset = dataset
        self.num_classes = num_classes

        dataset = split_vision_dataset(
            split=split,
            dataset=dataset,
            train_perc=self.train_perc,
            val_perc=self.val_perc,
            test_perc=self.test_perc,
            has_splits=has_splits,
        )

        mean, std = calculate_dataset_stats(dataset)
        print(f"Dataset statistics of split '{split}': mean={mean}, std={std}")

        return dataset, mean, std

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                train_split, train_mean, train_std = self._get_dataset_split(
                    split="train"
                )
                self.log_hyperparams(
                    {"fit_train_split_stats": {"mean": train_mean, "std": train_std}}
                )
                self.train_split = AEDataset(
                    dataset=train_split,
                    transform=self.default_transforms(
                        mean=train_mean, std=train_std, compose=True
                    ),
                )
                val_split, val_mean, val_std = self._get_dataset_split(split="val")
                self.log_hyperparams(
                    {"fit_val_split_stats": {"mean": val_mean, "std": val_std}}
                )
                self.val_split = AEDataset(
                    dataset=val_split,
                    transform=self.default_transforms(
                        mean=val_mean, std=val_std, compose=True
                    ),
                )
            case "validation":
                val_split, val_mean, val_std = self._get_dataset_split(split="val")
                self.log_hyperparams(
                    {"val_split_stats": {"mean": val_mean, "std": val_std}}
                )
                self.val_split = AEDataset(
                    dataset=val_split,
                    transform=self.default_transforms(
                        mean=val_mean, std=val_std, compose=True
                    ),
                )
            case "test":
                test_split, test_mean, test_std = self._get_dataset_split(split="test")
                self.log_hyperparams(
                    {"test_split_stats": {"mean": test_mean, "std": test_std}}
                )
                self.test_split = AEDataset(
                    dataset=test_split,
                    transform=self.default_transforms(
                        mean=test_mean, std=test_std, compose=True
                    ),
                )
            case "predict":
                # Predict from the test split
                predict_split, predict_mean, predict_std = self._get_dataset_split(
                    split="test"
                )
                self.log_hyperparams(
                    {"predict_split_stats": {"mean": predict_mean, "std": predict_std}}
                )
                self.predict_split = AEDataset(
                    dataset=predict_split,
                    transform=self.default_transforms(
                        mean=predict_mean, std=predict_std, compose=True
                    ),
                )
            case _:
                raise ValueError(f"Unsupported stage: '{stage}'")

    def teardown(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train_split = None
                self.val_split = None
            case "validation":
                self.val_split = None
            case "test":
                self.test_split = None
            case "predict":
                self.predict_split = None
            case _:
                raise ValueError(f"Unsupported stage: '{stage}'")

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_split,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batch_size,
        )
        print_dataloader_summary(dataloader=train_dataloader, split="train")
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_split,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=self.batch_size,
        )
        print_dataloader_summary(dataloader=val_dataloader, split="val")
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        test_dataloader = DataLoader(
            dataset=self.test_split,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=self.batch_size,
        )
        print_dataloader_summary(dataloader=test_dataloader, split="test")
        return test_dataloader

    def predict_dataloader(self) -> DataLoader:
        predict_dataloader = DataLoader(
            dataset=self.predict_split,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=self.batch_size,
        )
        print_dataloader_summary(dataloader=predict_dataloader, split="predict")
        return predict_dataloader
