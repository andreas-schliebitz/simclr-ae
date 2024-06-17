import uuid
import argparse

from argparse import Namespace


def get_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Unsupervised and Self-Supervised image feature extraction."
    )


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--devices",
        type=str,
        default="all",
        help="Comma separated list of device (GPUs) to use or '-1' for all.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data",
        help="Download path location of dataset to use.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[
            "Caltech101",
            "Caltech256",
            "CIFAR10",
            "CIFAR100",
            "FashionMNIST",
            "MNIST",
            "EMNIST",
            "STL10",
            "Imagenette",
            "DTD",
            "Country211",
            "Food101",
            "EuroSAT",
            "FGVCAircraft",
            "Flowers102",
            "GTSRB",
            "OxfordIIITPet",
        ],
        default="ImageNet",
        help="Dataset name to use for training and evaluation.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Square image size in pixels.",
    )
    parser.add_argument(
        "--train-perc",
        type=float,
        default=0.8,
        help="Train split percentage",
    )
    parser.add_argument(
        "--val-perc",
        type=float,
        default=0.1,
        help="Validation split percentage",
    )
    parser.add_argument(
        "--test-perc",
        type=float,
        default=0.2,
        help="Test split percentage",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--standardize",
        default=False,
        action="store_const",
        const=True,
        help="Standardize dataset splits using the Z-score normalization.",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=3,
        help="Number of image channels (RGB: 3, L: 1).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimization algorithm used while training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="ReduceLROnPlateau",
        choices=["ReduceLROnPlateau", "StepLR", "ExponentialLR", "CosineAnnealingLR"],
        help="Learning rate scheduler used while training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used while training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--latent-dimensions",
        type=int,
        default=512,
        help="Number of dimensions in latent space.",
    )
    parser.add_argument(
        "--lr-scheduler-gamma",
        type=float,
        default=None,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=40,
        help="Period of learning rate decay used while training with StepLR learning rate scheduler.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Momentum for SGD optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=0,
        help="Save top k models models only instead of all or the last.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs to wait for validation loss to improve.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory used for logging.",
    )
    parser.add_argument(
        "--use-mlflow",
        default=False,
        action="store_const",
        const=True,
        help="Use MLFlow for experiment tracking.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=str(uuid.uuid4().hex),
        help="Name used for the run",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="train",
        help="Name used for experiment.",
    )
    return parser


def get_args() -> Namespace:
    parser = get_parser()
    parser = add_args(parser)
    return parser.parse_args()
