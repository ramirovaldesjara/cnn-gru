import torch
import torch.optim as optim
from data_preparation import load_mesh_data, obtain_temporal_data
import torch.nn as nn

# # Load data
# pts_file = 'daltorso.pts'
# fac_file = 'daltorso.fac'
# points, edges, faces = load_mesh_data(pts_file, fac_file)
#
# # Create temporal data
# data_file = 'case0001.dat'
# data = obtain_temporal_data(data_file)
#
# # Define model
# spatial_feature_size = 32
# hidden_dim = 64
# num_gru_layers = 2
# model = HybridMeshCNN_GRU(spatial_feature_size, hidden_dim, num_gru_layers)
#
# # Training settings
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
#
# # Training loop
# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model(data.unsqueeze(0))
#     loss = criterion(output, data)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
