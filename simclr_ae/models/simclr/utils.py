import torch.nn as nn

from enum import Enum
from pathlib import Path
from pprint import pprint
from simclr_ae.utils import load_yaml
from simclr_ae.models.ae.model import AE
from simclr_ae.models.ae.module import AEModule


class ProjectionHead(str, Enum):
    MLP = "mlp"
    AE = "ae"


def load_projection_head_metadata(
    weights_path: str | Path,
) -> tuple[Path, dict]:
    weights_path = Path(weights_path).expanduser().resolve()
    print(f"Loading projection head weights from '{weights_path}'.")
    hparams_path = weights_path.parent.parent.joinpath("hparams.yaml")
    hparams = load_yaml(hparams_path)
    print(f"Loaded projection head hparams from '{hparams_path}':")
    pprint(hparams)
    return weights_path, hparams


def load_pretrained_encoder_projection_head(
    weights: Path,
    activation: nn.Module,
    num_channels: int,
    img_size: int,
    encoder_latent_dim: int,
    latent_dim: int,
    freeze: bool,
) -> nn.Sequential:
    # Autoencoder with input_dim used in training to allow loading of weights
    ae_model = AE(
        input_dim=(num_channels, img_size, img_size),
        latent_dim=encoder_latent_dim,
    )

    # Load the trained Autoencoder weights from checkpoint
    trained_ae_module = AEModule.load_from_checkpoint(
        checkpoint_path=weights, model=ae_model
    )

    # Extract trained Autoencoder embedding.
    # This is essentially a nn.Linear(512, encoder_latent_dim) layer:
    trained_embedding: nn.Linear = trained_ae_module.model.encoder.embedding
    # This should not have any effect since nn.Linear does not contain BatchNorm nor Dropout.
    trained_embedding.eval()

    if freeze:
        trained_embedding.requires_grad_(False)

    # Build projection head with pretrained and potentially frozen embedding layer.
    projection_head = nn.Sequential(
        trained_embedding,
        activation(),
        nn.Linear(encoder_latent_dim, latent_dim),
    )
    return projection_head


def create_simclr_model(
    model: nn.Module,
    batch_size: int,
    latent_dim: int,
    num_channels: int,
    img_size: int,
    projection_head_type: ProjectionHead = ProjectionHead.MLP,
    projection_head_activation: nn.Module = nn.ReLU,
    projection_head_embedding_dim: int = 512,
    projection_head_weights: Path = None,
    freeze_projection_head: bool = False,
) -> tuple[nn.Sequential, nn.Linear | nn.Sequential, dict]:
    """Creates the backbone (f) and projection head (g) as required by the SimCLR framework.
    The projection head can be the vanilla MLP projector or an MLP projector that uses
    a pretrained Autoencoder embedding as its input layer.

    Args:
        model (nn.Module): The backbone model used for feature extraction.
        batch_size (int): Batch size used for training.
        latent_dim (int): The desired number of output features in the output layer of the projector.
        num_channels (int): Number of channels (e.g. RGB=3) in an input image.
        img_size (int): Square size of input image.
        projection_head_type (ProjectionHead, optional): Type of projection head to use. Defaults to ProjectionHead.MLP.
        projection_head_activation (nn.Module, optional): Activation function to place between the input and output layer of the projector. Defaults to nn.ReLU.
        projection_head_weights (Path, optional): Filepath to the chckpoint of pretrained weights used for input layer of the projector.
        Only applicable for ProjectionHead.AE. Defaults to None.
        freeze_projection_head (bool, optional): Whether to allow the pretrained weights to be adjusted in training. Defaults to False.

    Raises:
        ValueError: Raised when an unsupported projector type is requested.

    Returns:
        tuple[nn.Sequential, nn.Linear | nn.Sequential, dict]: Backbone network, projection head, metadata about the projection head
    """
    model_layers = list(model.children())
    backbone_layers, output_layer = model_layers[:-1], model_layers[-1]

    num_features = output_layer.in_features
    print("Backbone output features:", num_features)

    if projection_head_weights:
        projection_head_weights, projection_head_hparams = (
            load_projection_head_metadata(weights_path=projection_head_weights)
        )
        projection_head_embedding_dim = projection_head_hparams["latent_dimensions"]

    projection_head_params = {
        "projection_head_embedding_dim": projection_head_embedding_dim
    }

    match projection_head_type:
        case ProjectionHead.MLP:
            # The default MLP projector will always have
            # an input layer of (512, 512) and
            # an output layer of (512, latent_dim={32,64,128})
            projection_head = nn.Sequential(
                nn.Linear(num_features, projection_head_embedding_dim),
                projection_head_activation(),
                nn.Linear(projection_head_embedding_dim, latent_dim),
            )
        case ProjectionHead.AE:
            if projection_head_weights:
                # The MLP projector with a pretrained Autoencoder embedding can have
                # an input layer of (512, projection_head_latent_dim={128,256,512}) and
                # an output layer of (projection_head_latent_dim={128,256,512}, latent_dim={32,64,128})
                # In this case 'projection_head_latent_dim' is the Autoencoder's 'latent_dim'.
                projection_head = load_pretrained_encoder_projection_head(
                    weights=projection_head_weights,
                    activation=projection_head_activation,
                    num_channels=num_channels,
                    img_size=img_size,
                    encoder_latent_dim=projection_head_embedding_dim,
                    latent_dim=latent_dim,
                    freeze=freeze_projection_head,
                )
            else:
                # Autoencoder compatible with backbone output dimensions
                projection_head = AE(
                    input_dim=(batch_size, num_features), latent_dim=latent_dim
                )

        case _:
            raise ValueError(
                f"Unsupported projection head type: {projection_head_type}"
            )

    print(
        f"Projection head '{projection_head_type}':",
        projection_head,
        list(projection_head.parameters(recurse=True)),
    )

    backbone = nn.Sequential(*backbone_layers)
    return backbone, projection_head, num_features, projection_head_params
