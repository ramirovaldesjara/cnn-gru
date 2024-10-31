import torch
import torch.nn as nn
from mesh_cnn import MeshCNNLayer
from temporal_GRU import TemporalGRU


class HybridModel(nn.Module):
    def __init__(self, num_nodes, time_steps, spatial_features, hidden_size):
        super(HybridModel, self).__init__()
        self.mesh_cnn = MeshCNNLayer(spatial_features, hidden_size)
        self.gru = TemporalGRU(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)  # Single output per node and time point

    def forward(self, x, edge_index, mask):
        # Step 1: Spatial feature extraction
        spatial_features = self.mesh_cnn(x, edge_index)

        # Step 2: Temporal modeling with GRU
        spatial_features = spatial_features.view(1, -1, spatial_features.size(-1))  # Reshape for GRU input
        temporal_features = self.gru(spatial_features).squeeze(0)  # Remove batch dimension

        # Step 3: Apply mask to predict missing values only
        out = self.output_layer(temporal_features)
        out = out * mask

        return out

