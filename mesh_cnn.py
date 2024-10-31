import torch
import torch.nn as nn


class MeshCNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MeshCNNLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.randn(out_features))  # Attention vector
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # Edge features: Euclidean distance (or other spatial features)
        edge_features = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)

        # Attention weights
        attention_scores = self.leaky_relu(self.a @ (self.W(x[edge_index[0]]) + self.W(x[edge_index[1]])))
        attention_weights = torch.softmax(attention_scores, dim=0)

        # Feature aggregation
        out = torch.zeros_like(x)
        out.index_add_(0, edge_index[0], attention_weights.unsqueeze(1) * edge_features.unsqueeze(1) * x[edge_index[1]])

        return out

