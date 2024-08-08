from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class Metric(ABC):

    @abstractmethod
    def apply(self, expected: NDArray, predicted: NDArray):
        pass


class RMSE(Metric):

    def apply(self, expected: NDArray, predicted: NDArray):
        return np.sqrt((expected - predicted).T.dot(expected - predicted) * (1 / expected.shape[0]))
