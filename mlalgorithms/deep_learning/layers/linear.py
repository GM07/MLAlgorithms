from mlalgorithms.deep_learning.deep_model import DeepModel, InferenceConfig

import torch
import torch.nn as nn

class Linear(DeepModel):
    """
    Layer which applies a linear layer to an input following the equation : y = xW + b

    The `b` term is optional
    """

    def __init__(self, input_size: int, output_size: int,  bias: bool = False) -> None:
        """
        
        input_size  :   Input size of the layer
        output_size :   Output size of the layer
        bias        :   Whether to use bias or not (`b` term in the equation y = Ax + b)

        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        # Create parameters for the weights and biases and make sure that requires_grad=True to 
        # keep gradients. 
        self.weights = nn.Parameter(torch.empty(self.input_size, self.output_size), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.output_size), requires_grad=True)
            
        self.initialize_parameters(self.input_size)

    def predict(self, X: torch.Tensor, config: InferenceConfig):
        assert X.shape[-1] == self.input_size, \
            f'The input shape of the given data ({X.shape[-1]}) is not compatible with the input size of the model ({self.input_size})'
        
        Y = X.to(config.device) @ self.weights.to(config.device)
        if self.use_bias:
            Y += self.bias.to(config.device)
        return Y
