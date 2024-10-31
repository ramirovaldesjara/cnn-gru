from train_model import *
from hybrid_model import *
from updated_train_model import *

if __name__ == '__main__':
    hola = 1
    # train_model()

    # Example data
    data = {
        'x': torch.randn(300, 300),  # Input data (300 nodes, 300 time points)
        'target': torch.randn(300, 300),  # Ground truth
    }
    edge_index = torch.randint(0, 300, (2, 500))  # Example edges in COO format
    mask = (torch.rand(300, 300) > 0.5).float()  # Random mask for missing values

    # Initialize and train model
    model = HybridModel(num_nodes=300, time_steps=300, spatial_features=300, hidden_size=64)
    train_model(model, data, edge_index, mask)
    evaluate_model(model, data, edge_index, mask)
