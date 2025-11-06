from abc import ABC, abstractmethod

import torch

class Metric(ABC):

    @abstractmethod
    def apply(self, expected: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, expected: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        return self.apply(expected, predicted)


class RMSE(Metric):

    def apply(self, expected: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        if expected.dim() == 1:
            expected = expected.unsqueeze(-1)
        if predicted.dim() == 1:
            predicted = predicted.unsqueeze(-1)
            
        return torch.sqrt((expected - predicted).T.matmul(expected - predicted) * (1 / expected.shape[0]))
