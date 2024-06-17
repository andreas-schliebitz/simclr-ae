import torch.nn as nn
import torchvision.transforms.functional as V

from torch import Tensor
from torch.utils.data import Dataset


class SimCLRDataset(Dataset):
    def __init__(
        self, dataset: Dataset, transform: nn.Module, augment: bool = True
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.augment = augment

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor | None, int | None]:
        dataset_item = self.dataset[index]
        if isinstance(dataset_item, tuple):
            image, label = dataset_item
        else:
            image = dataset_item
            label = None

        left: Tensor = self.transform(image)
        left_height, left_width = left.size()[-2:]

        image_tensor: Tensor = V.to_tensor(image)
        image_height, image_width = image_tensor.size()[-2:]

        if (left_height, left_width) != (image_height, image_width):
            image_tensor = V.resize(image_tensor, size=(left_height, left_width))

        if self.augment:
            right: Tensor = self.transform(image)
            return image_tensor, left, right, label

        return image_tensor, left, label

    def __len__(self) -> int:
        return len(self.dataset)
