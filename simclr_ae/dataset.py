import torch
import numpy as np
import torchvision.transforms.functional as V

from typing import Any
from torch import Tensor
from PIL.Image import Image
from joblib import Parallel, delayed
from simclr_ae.utils import RANDOM_SEED
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Subset, random_split


def get_vision_dataset(
    dataset_name: str,
    dataset_dir: str,
    split: str | None = None,
) -> tuple[Dataset, int, bool]:
    match dataset_name:
        case "Caltech101" | "Caltech256":
            CaltechDataset = getattr(datasets, dataset_name)
            dataset = CaltechDataset(
                root=dataset_dir,
                download=False,
                transform=transforms.Compose(
                    [transforms.Lambda(lambda image: image.convert("RGB"))]
                ),
            )
            has_splits = False
        case "CIFAR10" | "CIFAR100":
            CIFARDataset = getattr(datasets, dataset_name)
            dataset = CIFARDataset(
                root=dataset_dir,
                train=split in {"train", "val"},
                download=False,
            )
            has_splits = True
        case "FashionMNIST" | "MNIST":
            MNISTDataset = getattr(datasets, dataset_name)
            dataset = MNISTDataset(
                root=dataset_dir,
                train=split in {"train", "val"},
                download=False,
                transform=transforms.Compose(
                    [transforms.Lambda(lambda image: image.convert("RGB"))]
                ),
            )
            has_splits = True
        case "Imagenette":
            dataset = datasets.Imagenette(
                root=dataset_dir,
                split="train" if split == "train" else "val",
                size="full",
                download=False,
            )
            has_splits = True
        case "STL10":
            dataset = datasets.STL10(
                root=dataset_dir,
                split="train" if split in {"train", "val"} else "test",
                download=False,
            )
            has_splits = True
        case "EMNIST":
            dataset = datasets.EMNIST(
                root=dataset_dir,
                split="byclass",
                download=False,
                transform=transforms.Compose(
                    [transforms.Lambda(lambda image: image.convert("RGB"))]
                ),
            )
            has_splits = True
        case "DTD":
            dataset = datasets.DTD(
                root=dataset_dir,
                split="train" if split in {"train", "val"} else "test",
                download=False,
            )
            has_splits = True
        case "Country211":
            dataset = datasets.Country211(
                root=dataset_dir,
                split="train" if split in {"train", "val"} else "test",
                download=False,
            )
            has_splits = True
        case "Food101":
            dataset = datasets.Food101(
                root=dataset_dir,
                split="train" if split in {"train", "val"} else "test",
                download=False,
            )
            has_splits = True
        case "EuroSAT":
            dataset = datasets.EuroSAT(
                root=dataset_dir,
                download=False,
            )
            has_splits = False
        case "FGVCAircraft":

            def _remove_copyright_banner(image: Image) -> Image:
                width, height = image.size
                return image.crop((0, 0, width, height - 20))

            dataset = datasets.FGVCAircraft(
                root=dataset_dir,
                split="trainval" if split in {"train", "val"} else "test",
                annotation_level="manufacturer",
                download=False,
                transform=transforms.Compose(
                    [transforms.Lambda(_remove_copyright_banner)]
                ),
            )
            has_splits = True
        case "Flowers102":
            dataset = datasets.Flowers102(
                root=dataset_dir,
                split="train" if split in {"train", "val"} else "test",
                download=False,
            )
            has_splits = True
        case "GTSRB":
            dataset = datasets.GTSRB(
                root=dataset_dir,
                split="train" if split in {"train", "val"} else "test",
                download=False,
            )
            has_splits = True
        case "OxfordIIITPet":
            dataset = datasets.OxfordIIITPet(
                root=dataset_dir,
                split="trainval" if split in {"train", "val"} else "test",
                target_types="category",
                download=False,
            )
            has_splits = True
        case _:
            raise ValueError(f"Unsupported dataset name: '{dataset_name}'")

    return dataset, get_num_classes(dataset), has_splits


def split_vision_dataset(
    split: str,
    dataset: Dataset,
    train_perc: float,
    val_perc: float,
    test_perc: float,
    has_splits: bool,
) -> Dataset:
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    if has_splits:
        match split:
            case "train" | "val":
                train_split, val_split = random_split(
                    dataset,
                    lengths=[1 - val_perc, val_perc],
                    generator=generator,
                )
                return train_split if split == "train" else val_split
            case "test" | "predict":
                return dataset
            case _:
                raise ValueError(f"Unsupported split: '{split}'")
    else:
        train_split, val_split, test_split = random_split(
            dataset=dataset,
            lengths=[train_perc, val_perc, test_perc],
            generator=generator,
        )
        match split:
            case "train":
                return train_split
            case "val":
                return val_split
            case "test" | "predict":
                return test_split
            case _:
                raise ValueError(f"Unsupported split: '{split}'")


def get_num_classes(dataset: Dataset) -> int:
    if hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, "_labels"):
        num_classes = len(set(dataset._labels))
    elif hasattr(dataset, "y"):
        num_classes = len(set(dataset.y))
    elif isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
        num_classes = len(set(label for _, label in dataset))
    else:
        raise RuntimeError("Unable to determine number of classes for chosen dataset.")

    print("Number of classes in dataset:", num_classes)
    return num_classes


def calculate_dataset_stats(
    dataset: Dataset,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    def _calculate_mean_std(sample: Any) -> Tensor:
        sample = V.to_tensor(sample)
        mean = sample.mean(axis=(1, 2))
        std = sample.std(axis=(1, 2))
        return torch.cat((mean, std))

    print("Calculating dataset statistics (mean and standard deviation)...")

    mean_std: list[Tensor] = torch.stack(
        Parallel(n_jobs=-1)(
            delayed(_calculate_mean_std)(sample) for sample, _ in dataset
        )
    ).mean(dim=0, dtype=float)

    mean, std = mean_std[:3].numpy(), mean_std[-3:].numpy()
    return mean, std
