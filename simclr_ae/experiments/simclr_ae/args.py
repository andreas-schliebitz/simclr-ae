import argparse

from argparse import Namespace


def get_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Experiment for Self-Supervised image feature extraction using SimCLR with pretrained VAE als projection head."
    )


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        default="resnet18",
        help="Backbone architecture used for feature extraction.",
    )
    parser.add_argument(
        "--projection-head",
        type=str,
        choices=["mlp", "ae"],
        default="linear",
        help="Type of projection head.",
    )
    parser.add_argument(
        "--projection-head-activation",
        type=str,
        choices=[
            "ReLU",
            "LeakyReLU",
            "ELU",
            "GELU",
            "CELU",
            "SELU",
            "SiLU",
            "GLU",
            "Mish",
            "LogSigmoid",
            "Sigmoid",
            "Tanh",
        ],
        default="ReLU",
        help="Type of activation function to be used for projection head.",
    )
    parser.add_argument(
        "--projection-head-weights",
        type=str,
        default=None,
        help="Filepath to pretrained projection head weights.",
    )
    parser.add_argument(
        "--projection-head-embedding-dim",
        type=int,
        default=512,
        help="Number of output features within the MLPs input layer.",
    )
    parser.add_argument(
        "--freeze-projection-head",
        default=False,
        action="store_const",
        const=True,
        help="Freeze layers of projection head.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=["DEFAULT", "IMAGENET1K_V1"],
        default=None,
        help="Pretrained weights to use with ResNet18 model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature parameter for contrastive loss function.",
    )
    parser.add_argument(
        "--eval-train-perc",
        type=float,
        default=0.7,
        help="Percentage from evaluation dataset to use for training.",
    )
    parser.add_argument(
        "--eval-val-perc",
        type=float,
        default=0.1,
        help="Percentage from evaluation dataset to use for validation.",
    )
    parser.add_argument(
        "--eval-test-perc",
        type=float,
        default=0.2,
        help="Percentage from evaluation dataset to use for testing.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=100,
        help="Number of epochs to train evaluation procedure for.",
    )
    parser.add_argument(
        "--eval-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used while training evaluation procedure.",
    )
    parser.add_argument(
        "--eval-weight-decay",
        type=int,
        default=0,
        help="Weight decay used while training evalutation procedure.",
    )
    parser.add_argument(
        "--eval-patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs to wait for validation loss to improve while evaluating trained SimCLR model.",
    )
    parser.add_argument(
        "--experiment-eval-name",
        type=str,
        default="eval",
        help="Name used for evaluation experiment.",
    )
    return parser


def get_args() -> Namespace:
    parser = get_parser()
    parser = add_args(parser)
    return parser.parse_args()
