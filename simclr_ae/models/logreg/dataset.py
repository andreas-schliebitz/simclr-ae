from torch import Tensor
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, predictions: list[tuple[Tensor, Tensor]]) -> None:
        self.data = self._zip(predictions)
        self.latent_dim = len(self.data[0][0])

    def _zip(
        self, predictions: list[tuple[Tensor, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        data = []
        for features, labels in predictions:
            data.extend(zip(features.squeeze(), labels))
        return data

    def __getitem__(self, index: int) -> Tensor:
        feature, label = self.data[index]
        return feature, label

    def __len__(self) -> int:
        return len(self.data)
