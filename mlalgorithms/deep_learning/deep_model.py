from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum
from typing import Any

import torch
import torch.optim as O
import torch.nn as nn
import torch.utils.data as Data

from tqdm import tqdm

@dataclass
class InferenceConfig:
    device: str
    batch_size: int = 1

@dataclass
class TrainingConfig(InferenceConfig):
    epochs: int = 100
    lr: float = 0.01
    # TODO : Add support for multiple optimizers and loss functions
    optimizer = O.Adam
    loss_function = nn.CrossEntropyLoss

class DeepModel(nn.Module):

    def initialize_parameters(self, model_dimension):
        """
        Initializes the parameters of the model using an uniform distribution
        """
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/(model_dimension)) ** 0.5, b=(1 / (model_dimension)) ** 0.5)


    def fit(self, X: torch.Tensor, Y: torch.Tensor, config: TrainingConfig):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        tensor containing the features of all samples
        
        Y   :   (nb_samples,)
        tensor containing the prediction for all samples
        """
        dataset = self.tensors_to_dataset(X, Y)
        training_loader = Data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = config.optimizer(self.parameters(), lr=config.lr)
        loss_function = config.loss_function()

        self.to(config.device)
        for epoch in tqdm(range(config.epochs), total=config.epochs):
            self.train()
            for inputs, labels in training_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                
                optimizer.zero_grad()
                predictions = self.predict(inputs, config=config)
                loss = loss_function(predictions, labels)
                loss.backward()
                optimizer.step()

            # TODO : Evaluation
            # self.eval()
            # if epoch % 20 == 0:
                # print('Loss : ', loss.item())

        return self

    @abstractmethod
    def predict(self, X: torch.Tensor, config: InferenceConfig = None):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        pass

    @staticmethod
    def tensors_to_dataset(input: torch.Tensor, output: torch.Tensor) -> Data.Dataset:
        return Data.TensorDataset(input, output)
