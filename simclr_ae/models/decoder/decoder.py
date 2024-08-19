from torch import Tensor
from pythae.models.nn.default_architectures import Decoder_AE_MLP


class Decoder(Decoder_AE_MLP):
    def __init__(self, args: dict) -> None:
        super(Decoder, self).__init__(args)

    def forward(self, z: Tensor, output_layer_levels: list[int] = None) -> Tensor:
        return super(Decoder, self).forward(
            z=z, output_layer_levels=output_layer_levels
        )["reconstruction"]
