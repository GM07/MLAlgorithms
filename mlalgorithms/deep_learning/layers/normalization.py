from typing import Union
from mlalgorithms.deep_learning.deep_model import DeepModel, InferenceConfig

import torch
import torch.nn as nn

class LayerNormalization(DeepModel):
    """
    Applies Layer Normalization, with rescaling and shift, only on the last D dimensions.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        normalized_shape    :   Size of last D dimensions onto which the layer normalization
                                will be applied. For a tensor of size (B, C, F, F), if we want to perform layer 
                                normalization over the last 2 layers, we would set normalized_shape = (F, F)
        eps                 :   Epsilon value added in case the variance is null
        """

        super(LayerNormalization, self).__init__()
        self.normalized_shape = normalized_shape
        print(self.normalized_shape)
        if isinstance(self.normalized_shape, int):
            self.normalized_shape = (self.normalized_shape, )
        self.dims = tuple(range(-len(self.normalized_shape), 0))
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def predict(self, X: torch.Tensor, config: InferenceConfig):
        means = X.mean(dim=self.dims, keepdim=True)
        vars = X.var(dim=self.dims, unbiased=False, keepdim=True)
        normalized_X = (X - means) / torch.sqrt(vars + self.eps)
        return normalized_X * self.weight + self.bias

class BatchNormalization(DeepModel):
    """
    Applies Batch Normalization, with rescaling and shift. This class works for 1d batch norm and 2d batch norm
    
    Based on the paper : "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"

    Args :
    channel_dimension   :   Index of dimension where batch normalization will be applied
    channel_size        :   Size of the channel dimension
    
    """

    def __init__(self, channel_dimension, channel_size, eps=1e-5):
        super(BatchNormalization, self).__init__()
        self.channel_dimension = channel_dimension
        self.channel_size = channel_size
        self.eps = eps
        self.momentum = 1.0

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(channel_size))
        self.bias = nn.Parameter(torch.zeros(channel_size))

        # Buffers are persistent and will be saved alongside parameters. 
        # However, they are not learnable.
        self.register_buffer('running_mean', torch.zeros(channel_size))
        self.register_buffer('running_var', torch.ones(channel_size))

    def predict(self, X: torch.Tensor, config: InferenceConfig):
        nb_dims = X.dim()
        stats_dims = list(range(nb_dims))
        del stats_dims[self.channel_dimension]
        stats_view = [1] * (nb_dims)
        stats_view[self.channel_dimension] = -1

        if self.training:
            # Calculate batch statistics
            batch_mean = X.mean(dim=tuple(stats_dims))
            batch_var = X.var(dim=tuple(stats_dims), unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.view(*stats_view)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.view(*stats_view)

            # Normalize using batch statistics
            X_normalized = (X - batch_mean.view(*stats_view)) / torch.sqrt(batch_var.view(*stats_view) + self.eps)
        else:
            # Normalize using running statistics
            X_normalized = (X - self.running_mean.view(*stats_view)) / torch.sqrt(self.running_var.view(*stats_view) + self.eps)

        # Apply learnable parameters
        return (self.weight.view(*stats_view) * X_normalized + self.bias.view(*stats_view))


    def initialize_parameters(self, model_dimension):
        """
        Initializes the parameters of the model
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
