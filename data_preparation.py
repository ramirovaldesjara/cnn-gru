import numpy as np
import torch
import os

def load_mesh_data(pts_file, fac_file):
    # Load points and faces
    points = load_file(pts_file)  # Shape: (num_nodes, 3)
    faces = load_file(fac_file)  # Shape: (num_faces, 3)

    # Generate edges from faces
    edges = []
    for face in faces:
        edges.append((face[0], face[1]))
        edges.append((face[1], face[2]))
        edges.append((face[2], face[0]))
    edges = list(set(edges))  # Remove duplicates

    return points, edges, faces


def obtain_temporal_data(data_file, feature_dim=1):

    time_series_data = get_filtered_data(data_file)

    return torch.tensor(time_series_data, dtype=torch.float32)

def get_filtered_data(file_name):
    # Add the 'data' directory to the file path
    file_name = os.path.join('data', file_name)

    # Initialize a list to store the filtered data
    filtered_data = []

    # Read the file line by line
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Start processing after the first four lines
    start_processing = False
    for i, line in enumerate(lines):
        # Skip the first four lines
        if i < 4:
            continue

        # Split the line into components
        split_line = line.strip().split()

        # Ensure the line has enough columns to contain node data
        if len(split_line) >= 357:  # At least 5 columns for sample, time, flags, limb leads + 352 node columns
            # Extract the last 352 columns, which are the node data
            node_data = list(map(float, split_line[-352:]))
            filtered_data.append(node_data)

    # Convert the filtered data to a NumPy array for easier manipulation if needed
    filtered_data_array = np.array(filtered_data)
    return filtered_data_array


def load_file(file_name):
    # Add the 'data' directory to the file path
    file_name = os.path.join('data', file_name)

    # Initialize an empty list to store the coordinates
    coordinates = []

    # Open the .pts file and read the lines
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split each line by spaces and convert to float
            coords = list(map(float, line.strip().split()))
            # Append the coordinates as a list to the main list
            coordinates.append(coords)

    # Convert the list of lists into a NumPy array
    coordinates_array = np.array(coordinates)

    return coordinates_array
