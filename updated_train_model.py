import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hybrid_model import HybridModel


def train_model(model, data, edge_index, mask, epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data['x'], edge_index, mask)

        # Compute loss only on observed values
        loss = masked_mse_loss(output, data['target'], mask)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")

def masked_mse_loss(predicted, target, mask):
    mse_loss = (predicted - target) ** 2
    masked_loss = mse_loss * mask
    return masked_loss.sum() / mask.sum()

def evaluate_model(model, data, edge_index, mask):
    model.eval()
    with torch.no_grad():
        output = model(data['x'], edge_index, mask)
        rpe = torch.norm(data['target'] - output) / torch.norm(data['target'])
        print(f"Relative Prediction Error (RPE): {rpe.item()}")


