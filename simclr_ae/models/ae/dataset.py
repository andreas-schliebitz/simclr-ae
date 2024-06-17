import torch.nn as nn

from torch import Tensor
from torch.utils.data import Dataset


class AEDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: nn.Module) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Tensor, int | None]:
        dataset_item = self.dataset[index]
        if isinstance(dataset_item, tuple):
            image, label = dataset_item
        else:
            image = dataset_item
            label = None

        return self.transform(image), label

    def __len__(self) -> int:
        return len(self.dataset)
