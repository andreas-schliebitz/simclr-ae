import torch.nn as nn

from torch import Tensor
from pythae.models import BaseAEConfig
from simclr_ae.models.encoder.encoder import Encoder
from simclr_ae.models.decoder.decoder import Decoder


class AE(nn.Module):
    def __init__(self, input_dim: tuple, latent_dim: int) -> None:
        super(AE, self).__init__()
        model_config = BaseAEConfig(input_dim=input_dim, latent_dim=latent_dim)
        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        r = self.decoder(h)
        return r
