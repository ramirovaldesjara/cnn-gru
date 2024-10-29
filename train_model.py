import torch
import torch.optim as optim
from data_preparation import load_mesh_data, create_temporal_data
from hybrid_model import HybridMeshCNN_GRU
import torch.nn as nn

# Load data
pts_file = 'path/to/pts_file.pts'
fac_file = 'path/to/fac_file.fac'
points, edges, faces = load_mesh_data(pts_file, fac_file)

# Create temporal data
num_timepoints = 100
num_nodes = points.shape[0]
data = create_temporal_data(num_timepoints, num_nodes)

# Define model
spatial_feature_size = 32
hidden_dim = 64
num_gru_layers = 2
model = HybridMeshCNN_GRU(spatial_feature_size, hidden_dim, num_gru_layers)

# Training settings
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # For example, for predicting missing values

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data.unsqueeze(0))  # Add batch dimension if needed
    loss = criterion(output, data)  # Compare output to original data
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
