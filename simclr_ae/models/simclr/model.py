import torch.nn as nn

from torch import Tensor


class SimCLRModel(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module) -> None:
        super(SimCLRModel, self).__init__()
        self.f = f
        self.g = g

    def forward(self, x_i: Tensor, x_j: Tensor) -> tuple[Tensor, Tensor]:
        def _project(x: Tensor) -> Tensor:
            batch_size = x.size(0)
            h = self.f(x).view(batch_size, -1)
            z = self.g(h)
            return z

        z_i = _project(x_i)
        z_j = _project(x_j)

        return z_i, z_j
