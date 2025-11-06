from mlalgorithms.deep_learning.deep_model import DeepModel, InferenceConfig

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(DeepModel):
    def __init__(self, head_size, num_heads, *args, **kwargs):
        super(DeepModel, self).__init__(*args, **kwargs)
        self.head_size = head_size
        self.num_heads = num_heads

        self.mask_softmax = float('-inf')
        self.w_q = nn.Parameter(torch.empty(self.num_heads * self.head_size, self.num_heads * self.head_size))
        self.w_k = nn.Parameter(torch.empty(self.num_heads * self.head_size, self.num_heads * self.head_size))
        self.w_v = nn.Parameter(torch.empty(self.num_heads * self.head_size, self.num_heads * self.head_size))
        self.w_o = nn.Parameter(torch.empty(self.num_heads * self.head_size, self.num_heads * self.head_size))

        self.b_q = nn.Parameter(torch.empty(self.num_heads * head_size))
        self.b_k = nn.Parameter(torch.empty(self.num_heads * head_size))
        self.b_v = nn.Parameter(torch.empty(self.num_heads * head_size))
        self.b_o = nn.Parameter(torch.empty(self.num_heads * head_size))

        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/(self.head_size * self.num_heads))**0.5, b=(1/(self.head_size * self.num_heads))**0.5)


    def get_attention_weights(self, queries, keys, mask=None):
        """Computes the attention weights"""
        weights = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size) 
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) # (sequence_length, 1)
            weights = weights.masked_fill(mask == 0, self.mask_softmax) # Fill in small value to cancel softmax
        
        weights = F.softmax(weights, dim=-1)
        return weights
        
    def apply_attention(self, queries, keys, values, mask=None):
        """Applies the attention"""
        attention_weights = self.get_attention_weights(queries=queries, keys=keys, mask=mask)
        attented_values = attention_weights @ values
        return self.merge_heads(attented_values)

    def split_heads(self, tensor: torch.Tensor):
        """Split all vectors between heads"""
        batch_size, seq_len, _ = tensor.shape
        inputs = tensor.view(batch_size, seq_len, self.num_heads, -1)
        inputs = inputs.transpose(1, 2)
        return inputs
        
    def merge_heads(self, tensor: torch.FloatTensor):
        """Merge all head vectors"""
        batch_size, _, seq_len, _ = tensor.shape
        outs = tensor.transpose(1, 2)
        return outs.reshape(batch_size, seq_len, -1)

    def predict(self, X: torch.Tensor, config: InferenceConfig):
        """Applies Multi-headed attention to input X. X must be a tensor of size
        [3, batch_size, sequence_length, num_heads * head_size] where the first 
        dimension is indexed in the following way : [queries, keys, values]
        """
        Q = X[0] @ self.w_q + self.b_q
        K = X[1] @ self.w_k + self.b_k
        V = X[2] @ self.w_v + self.b_v

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        Y = self.apply_attention(Q, K, V, mask=config.mask) # (batch_size, sequence_length, num_heads * head_size)
        return Y @ self.w_o + self.b_o
