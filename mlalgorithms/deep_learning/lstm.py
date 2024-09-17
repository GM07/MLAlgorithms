from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlalgorithms.deep_learning.deep_model import DeepModel, InferenceConfig


class LSTM(DeepModel):
    """
    Long-Short Term Memory model based on the paper "LONG SHORT-TERM MEMORY" (https://www.bioinf.jku.at/publications/older/2604.pdf)
    """

    def __init__(self, input_size, hidden_size):
        """
        input_size  :   Input size of model
        hidden_size :   Hidden size that will be used by the gates and cell states
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ii = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_if = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_ig = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_io = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ii = nn.Parameter(torch.empty(hidden_size))
        self.b_if = nn.Parameter(torch.empty(hidden_size))
        self.b_ig = nn.Parameter(torch.empty(hidden_size))
        self.b_io = nn.Parameter(torch.empty(hidden_size))

        self.w_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hg = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hi = nn.Parameter(torch.empty(hidden_size))
        self.b_hf = nn.Parameter(torch.empty(hidden_size))
        self.b_hg = nn.Parameter(torch.empty(hidden_size))
        self.b_ho = nn.Parameter(torch.empty(hidden_size))

        self.initialize_parameters(self.hidden_size)

    def initialize_parameters(self, model_dimension):
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1 / model_dimension) ** 0.5, b=(1 / model_dimension) ** 0.5)


    def predict(self, X: torch.Tensor, config: InferenceConfig = None):
        """
        Encodes the sequences of X into a vector of size `self.hidden_size`. We expect
        the config.init_hidden_size to have the initial state of the hidden states. The
        tensor must be a tensor of size (1, batch_size, hidden_size * 2). The first half
        of hidden_size * 2 will be used to initialize the hidden state and the second half
        will be used to initialize the cell state

        Args :
        -----------
        X   :   (nb_samples, seq_length, hidden_size)

        """
        h_t = config.init_hidden_size[0, :, :self.hidden_size]
        c_t = config.init_hidden_size[0, :, self.hidden_size:]

        self.forward(X, (h_t, c_t))

    def forward(self, X: torch.Tensor, hidden_states: Tuple[torch.Tensor, torch.Tensor]):
        """
        Args :
        -----------
        X               :   tensor of shape (batch_size, sequence_length, hidden_size) containing the embedded sequences.
        hidden_states   :   Tuple containing (h, c)
            - h : a tensor of shape `(1, batch_size, hidden_size) representing the initial hidden state
            - c : a tensor of shape `(1, batch_size, hidden_size) representing the initial cell state

        Returns
        -----------
        A tuple with :
            - A feature tensor of size (batch_size, sequence_length, hidden_size) encoding the input sentence. 
            - The final hidden state which is a tuple with (h, c)
                - h : a tensor of shape `(1, batch_size, hidden_size) representing the initial hidden state
                - c : a tensor of shape `(1, batch_size, hidden_size) representing the initial cell state

        """
        h_t = hidden_states[0]
        c_t = hidden_states[1]

        batch_size, seq_len, hidden_size = X.shape

        output = torch.zeros((seq_len, batch_size, self.hidden_size))

        for i in range(seq_len):
            current_input = X[:, i, :]

            # Gates
            i_t = F.sigmoid((current_input @ self.w_ii.T) + self.b_ii + (h_t @ self.w_hi.T) + self.b_hi)
            f_t = F.sigmoid((current_input @ self.w_if.T) + self.b_if + (h_t @ self.w_hf.T) + self.b_hf)
            g_t = F.tanh((current_input @ self.w_ig.T) + self.b_ig + (h_t @ self.w_hg.T) + self.b_hg)
            o_t = F.sigmoid((current_input @ self.w_io.T) + self.b_io + (h_t @ self.w_ho.T) + self.b_ho)

            # Cell state and hidden state
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * F.tanh(c_t)
            output[i, :, :] = h_t

        return output.transpose(0, 1), (h_t, c_t)


class UnidirectionalLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size = 256,
        hidden_size = 256,
    ):
        super(UnidirectionalLSTMEncoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.rnn = LSTM(input_size=embedding_size, hidden_size=hidden_size)

    def forward(self, inputs, hidden_states):
        """
        Encodes a sequence using a unidirectional encoder

        Args :
        -----------
        inputs          :   tensor of shape (batch_size, sequence_length, hidden_size) containing the embedded sequences.
        hidden_states   :   Tuple containing
            - h : a tensor of shape `(1, batch_size, hidden_size)
            - c : a tensor of shape `(1, batch_size, hidden_size)

        Returns
        -----------
        A tuple with :
            - A feature tensor of size (batch_size, sequence_length, hidden_size) encoding the input sentence. 
            - The final hidden state which is a tuple with
                - h a tensor of shape (1, batch_size, hidden_size)
                - c a tensor of shape (1, batch_size, hidden_size)
        """

        x = self.embedding(inputs)
        return self.rnn(x, hidden_states)

def initial_states(self, batch_size, device=None):
    if device is None:
        device = next(self.parameters()).device
    shape = (2, batch_size, self.hidden_size)
    h_0 = torch.zeros(shape, dtype=torch.float, device=device)
    return (h_0, h_0)
