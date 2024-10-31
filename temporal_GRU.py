import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TemporalGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemporalGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # GRU expects input of shape [batch, sequence, features]
        out, _ = self.gru(x)
        return out
