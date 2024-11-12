import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import torch
from torch_geometric.nn import GCNConv, BatchNorm
import torch.nn.functional as F
from torch.optim import Adam, SGD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

def load_data(filepath):
    """
    Load the data and split it into training and testing sets with a 3:7 ratio.

    Parameters:
    filepath (str): Path to the data file

    Returns:
    X_train, X_test, y_train, y_test: Features and target for training and testing sets
    """

    data = pd.read_csv(filepath)
    
    X = data.drop(columns=['Rings'])  
    y = data['Rings']
    
    
    return X,y

def create_graph_data(X, y, K=5):
    """
    Preprocess features and labels to create a PyTorch Geometric Data object.

    Parameters:
    - X: Feature DataFrame
    - y: Target Series
    - K: Number of neighbors for KNN adjacency matrix

    Returns:
    - graph_data: PyTorch Geometric Data object with x, edge_index, and y attributes
    - input_features: Number of input features for the model
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert y to a tensor
    y_adjusted=y-1
    y_tensor = torch.tensor(y_adjusted.values, dtype=torch.long)  # Use long for classification

    # Create adjacency matrix using KNN
    adjacency_matrix = kneighbors_graph(X_scaled, K, mode='connectivity', include_self=False).toarray()

    # Convert adjacency matrix to edge index format
    edge_index = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert X to a tensor for the features
    x_tensor = torch.tensor(X_scaled, dtype=torch.float)
    
    # Create graph data object
    graph_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    input_features = X_scaled.shape[1]
    
    return graph_data, input_features



# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_dim, dropout_rate, num_classes=4):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# 超参数网格搜索函数
def grid_search_gcn(graph_data, input_features, num_classes=4, epochs=100):
    # 定义超参数搜索空间
    param_grid = {
        'hidden_dim': [32, 64, 128],
        'dropout_rate': [0.3, 0.5, 0.7],
        'learning_rate': [0.01, 0.005, 0.001],
        'weight_decay': [5e-4, 1e-4, 1e-5]
    }

    # 定义用于存储最佳结果的变量
    best_accuracy = 0
    best_params = None

    # 遍历每个超参数组合
    for hidden_dim in param_grid['hidden_dim']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                for weight_decay in param_grid['weight_decay']:
                    print(f"\nTraining with hidden_dim={hidden_dim}, dropout_rate={dropout_rate}, "
                          f"learning_rate={learning_rate}, weight_decay={weight_decay}")
                    
                    # 初始化模型和优化器
                    model = GCN(input_features, hidden_dim, dropout_rate, num_classes)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    # 训练模型
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        out = model(graph_data)
                        loss = criterion(out, graph_data.y)
                        loss.backward()
                        optimizer.step()

                    # 评估模型
                    model.eval()
                    with torch.no_grad():
                        logits = model(graph_data)
                        predictions = logits.argmax(dim=1)
                        accuracy = accuracy_score(graph_data.y.cpu(), predictions.cpu())

                        print(f"Accuracy with current parameters: {accuracy * 100:.2f}%")
                        
                        # 检查当前超参数组合是否是最佳
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'hidden_dim': hidden_dim,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'weight_decay': weight_decay
                            }

    # 输出最佳超参数组合和对应的准确率
    print("\nBest Parameters:")
    print(f"Hidden Dimension: {best_params['hidden_dim']}")
    print(f"Dropout Rate: {best_params['dropout_rate']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Weight Decay: {best_params['weight_decay']}")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

# 假设已生成 graph_data 和 input_features
# grid_search_gcn(graph_data, input_features)
X,y = load_data('C:/Users/Admin/Desktop/NN_assignment3/data/abalone.csv')

# # Decision Tree
# print("Running Decision Tree Experiments")
# decision_tree_experiment_multiple(X, y, n_experiments=10)

# # Random Forest
# print("\nRunning Random Forest Experiments")
# random_forest_experiment_multiple(X, y, n_experiments=10)

# # Simple Neural Network 
# print("\nRunning Simple Neural Network Experiments")
# neural_network_experiment_multiple(X, y, n_experiments=10)

#GNN
print("\nRunning Graph Neural Network Experiments")
graph_data, input_features = create_graph_data(X, y)
grid_search_gcn(graph_data, input_features )