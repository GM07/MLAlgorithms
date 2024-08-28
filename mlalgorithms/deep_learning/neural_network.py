from mlalgorithms.deep_learning.deep_model import DeepModel, InferenceConfig

from typing import List

import torch
import torch.nn as nn

class NeuralNetwork(DeepModel):

    ACTIVATION_FUNCTIONS = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }

    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], hidden_activations: List[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sizes = [input_size] + hidden_sizes + [output_size]

        self.hidden_activations = hidden_activations
        self.final_activation = nn.Softmax(-1) if output_size > 2 else nn.Sigmoid()

        assert len(hidden_sizes) == len(self.hidden_activations), \
            f'The layer sizes and activations must be the same size but are {len(hidden_sizes)} vs {len(self.hidden_activations)}'

        assert all(map(lambda x: x in NeuralNetwork.ACTIVATION_FUNCTIONS, hidden_activations)), \
            f'One of the given activation function is not valid'

        self.m_hidden_activations = list(map(lambda x: NeuralNetwork.ACTIVATION_FUNCTIONS[x](), hidden_activations))

        # Note : ModuleList vs Sequential vs ParameterList
        # ModuleList : allows to store Module as a list which can be useful when iterating
        # through the layers is needed to store or use some information. ModuleList
        # does not have a forward layer. Usually used when the intermediate inputs are needed (U-Net)
        #
        # ParameterList : same as ModuleList, but for parameters (does not inherit from Module)
        #
        # Sequential : does have a forward method which means that all layers inside will be
        # connected

        self.shapes = []
        for i in range(len(self.sizes) - 1):
            self.shapes.append((self.sizes[i], self.sizes[i + 1]))

        # Create parameters for the weights and biases and make sure that requires_grad=True to 
        # keep gradients. 
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(in_shape, out_shape), requires_grad=True) for in_shape, out_shape in self.shapes])
        self.biases = nn.ParameterList([nn.Parameter(torch.empty(out_shape), requires_grad=True) for _, out_shape in self.shapes])
        self.activations = self.m_hidden_activations + [self.final_activation]

        self.initialize_parameters(max(self.sizes))

    def predict(self, X: torch.Tensor, config: InferenceConfig):
        assert X.shape[-1] == self.sizes[0], \
            f'The input shape of the given data ({X.shape[-1]}) is not compatible with the input size of the model ({self.sizes[0]})'
        
        y = X.to(config.device)
        for i in range(len(self.weights)):
            # Note : While the data property of a parameter gets the underlying tensor behind the parameter class, it does not
            # go through PyTorch autograd. Thus the results will not have valid gradients
            y = y @ self.weights[i].to(config.device) + self.biases[i].to(config.device)
            y = self.activations[i](y)
        return y
