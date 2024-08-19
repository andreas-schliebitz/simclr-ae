import os
import torch
import lightning as L

from simclr_ae.utils import RANDOM_SEED
from simclr_ae.utils import print_dataloader_summary
from torch.utils.data import DataLoader, Dataset, random_split


class LogisticRegressionDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_perc: float,
        val_perc: float,
        test_perc: float,
        batch_size: int = 32,
        num_workers: int = -1,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.train_perc = train_perc
        self.val_perc = val_perc
        self.test_perc = test_perc
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() if num_workers <= 0 else num_workers

        self.train_split = None
        self.val_split = None
        self.test_split = None

    def setup(self, stage: str) -> None:
        generator = torch.Generator()
        generator.manual_seed(RANDOM_SEED)

        split_lengths = [self.train_perc, self.val_perc, self.test_perc]
        train_split, val_split, test_split = random_split(
            self.dataset,
            lengths=split_lengths,
            generator=generator,
        )

        match stage:
            case "fit":
                self.train_split = train_split
                self.val_split = val_split
            case "validation":
                self.val_split = val_split
            case "test":
                self.test_split = test_split
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
