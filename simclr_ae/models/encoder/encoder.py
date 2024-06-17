from torch import Tensor
from pythae.models.nn.default_architectures import Encoder_AE_MLP


class Encoder(Encoder_AE_MLP):
    def __init__(self, args: dict) -> None:
        super(Encoder, self).__init__(args)

    def forward(self, x: Tensor, output_layer_levels: list[int] = None) -> Tensor:
        return (
            super(Encoder, self)
            .forward(x=x, output_layer_levels=output_layer_levels)
            .embedding
        )
