import argparse

from argparse import Namespace


def get_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Unsupervised training and evaluation of Autoencoder architectures."
    )


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Add AE specific arguments here
    return parser


def get_args() -> Namespace:
    parser = get_parser()
    parser = add_args(parser)
    return parser.parse_args()
