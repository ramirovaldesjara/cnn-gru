import numpy as np
import torch


def load_mesh_data(pts_file, fac_file):
    # Load points and faces
    points = np.loadtxt(pts_file)  # Shape: (num_nodes, 3)
    faces = np.loadtxt(fac_file, dtype=int)  # Shape: (num_faces, 3)

    # Generate edges from faces
    edges = []
    for face in faces:
        edges.append((face[0], face[1]))
        edges.append((face[1], face[2]))
        edges.append((face[2], face[0]))
    edges = list(set(edges))  # Remove duplicates

    return points, edges, faces


def create_temporal_data(num_timepoints, num_nodes, feature_dim=1):
    # Initialize tensor for storing time series data
    time_series_data = np.random.rand(num_timepoints, num_nodes, feature_dim)  # Example random data
    # Replace with actual loading logic if necessary
    return torch.tensor(time_series_data, dtype=torch.float32)
