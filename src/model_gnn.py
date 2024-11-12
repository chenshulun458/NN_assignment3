from ucimlrepo import fetch_ucirepo
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import pandas as pd

def load_and_preprocess_data():
    # Load data
    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features
    y = abalone.data.targets
    
    # Standardize features and encode categorical variables
    scaler = StandardScaler()
    X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
    X = scaler.fit_transform(X)
    
    # Convert labels to tensor format
    y = torch.tensor(y.values, dtype=torch.float)
    
    # Build adjacency matrix using KNN
    K = 5  # Number of neighbors
    adjacency_matrix = kneighbors_graph(X, K, mode='connectivity', include_self=False).toarray()
    
    # Convert adjacency matrix to edge index format (required by PyTorch Geometric)
    edge_index = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    
    # Convert labels to binary classification
    y_binary = (y >= 7).float()
    
    # Create graph data object
    graph_data = Data(x=x, edge_index=edge_index, y=y_binary.view(-1))
    return graph_data, X.shape[1]

class GCN(torch.nn.Module):
    def __init__(self, input_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_features, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

def train_model(graph_data, input_features, epochs=200, lr=0.01, weight_decay=5e-4):
    model = GCN(input_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = criterion(out, graph_data.y)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            model.eval()
            logits = model(graph_data)
            predictions = (logits >= 0).float()
            accuracy = (predictions == graph_data.y).sum().item() / len(graph_data.y)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')

# Load and preprocess data
graph_data, input_features = load_and_preprocess_data()

# Train the model
train_model(graph_data, input_features)
