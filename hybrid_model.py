import torch
import torch.nn as nn
from mesh_cnn import MeshConvLayer


class HybridMeshCNN_GRU(nn.Module):
    def __init__(self, spatial_feature_size, hidden_dim, num_gru_layers):
        super(HybridMeshCNN_GRU, self).__init__()
        # Define MeshCNN layer
        self.meshcnn = MeshConvLayer(in_channels=3, out_channels=spatial_feature_size)
        # Define GRU layer
        self.gru = nn.GRU(input_size=spatial_feature_size, hidden_size=hidden_dim, num_layers=num_gru_layers,
                          batch_first=True)
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, spatial_feature_size)

    def forward(self, x):
        batch_size, num_timepoints, num_nodes, node_features = x.size()

        spatial_features = []
        for t in range(num_timepoints):
            spatial_feature_t = self.meshcnn(x[:, t, :, :])  # Spatial features per timepoint
            spatial_features.append(spatial_feature_t)

        # Stack and pass through GRU
        spatial_features = torch.stack(spatial_features, dim=1)
        gru_output, _ = self.gru(spatial_features)

        # Final prediction layer
        predictions = self.output_layer(gru_output)
        return predictions
