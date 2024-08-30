from abc import ABC, abstractmethod

import torch

class Metric(ABC):

    @abstractmethod
    def apply(self, expected: torch.Tensor, predicted: torch.Tensor):
        pass

    def __call__(self, expected: torch.Tensor, predicted: torch.Tensor) -> torch.Any:
        return self.apply(expected, predicted)


class RMSE(Metric):

    def apply(self, expected: torch.Tensor, predicted: torch.Tensor):
        return torch.sqrt((expected - predicted).T.matmul(expected - predicted) * (1 / expected.shape[0]))
