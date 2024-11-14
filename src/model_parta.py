import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.optim import Adam, SGD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

# 1.load the data & split it into training and testing sets with a 3:7 ratio

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


# 2.train and evaluate model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    """Train and evaluate model using Accuracy and F1 Score"""
    # 训练模型
    model.fit(X_train, y_train.values.ravel())
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率和F1分数
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # 使用加权F1分数，以便在类别不均衡时有更合理的衡量
    
    return accuracy, f1

# 3.experiment visualization

# 4. n times experiment
def run_experiments(n_experiments, model_func, X, y, selected_features=None):
    accuracy_list = []

    for experiment_number in range(n_experiments):
        # 每次随机分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=experiment_number)

        # 调用模型函数
        accuracy,f1 = model_func(X_train, X_test, y_train, y_test)

        # 保存accuracy
        accuracy_list.append(accuracy)

        print(f'Experiment {experiment_number + 1} - Accuracy: {accuracy:.4f}')

    # give final results
    print("\nFinal Results after running multiple experiments:")
    print(f'Average Accuracy: {np.mean(accuracy_list):.4f}, Accuracy Std Dev: {np.std(accuracy_list):.4f}')



# 5. Decision Tree +
def decision_tree_experiment_multiple(X, y, selected_features=None, n_experiments=10):
    """决策树模型实验运行多次"""
    def model_func(X_train, X_test, y_train, y_test):
        # create a decision tree classifier
        dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2,criterion='entropy')
        # use the train_and_evaluate_model function to train and evaluate the model
        accuracy, f1 = train_and_evaluate_model(X_train, X_test, y_train, y_test, dt_clf)
        return accuracy, f1
    
    # run experiments
    run_experiments(n_experiments, model_func, X, y, selected_features)


# 6. Random Forest 
from sklearn.ensemble import RandomForestClassifier

def random_forest_experiment_multiple(X, y, selected_features=None, n_experiments=10):
    """Random Forest model experiment run multiple times"""
    def model_func(X_train, X_test, y_train, y_test):
        # Create a Random Forest classifier with specific hyperparameters
        rf_clf = RandomForestClassifier()
        # Use the train_and_evaluate_model function to train and evaluate the model
        accuracy, f1 = train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_clf)
        return accuracy, f1
    
    # Run multiple experiments
    run_experiments(n_experiments, model_func, X, y, selected_features)


# 7. Simple Neural Network
def neural_network_experiment_multiple(X, y, selected_features=None, n_experiments=10, optimizer='adam'):
    """Neural Network model experiment run multiple times using Adam or SGD optimizer."""
    def model_func(X_train, X_test, y_train, y_test):
        # Create a neural network classifier with specified optimizer and hyperparameters
        nn_clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver=optimizer, max_iter=200, random_state=42)
        # Use the train_and_evaluate_model function to train and evaluate the model
        accuracy, f1 = train_and_evaluate_model(X_train, X_test, y_train, y_test, nn_clf)
        return accuracy, f1
    
    # Run multiple experiments
    run_experiments(n_experiments, model_func, X, y, selected_features)

# 8. Graph Neural Network
def gcn_experiment_multiple(graph_data, input_features, num_classes=4, n_experiments=10, epochs=500, lr= 0.01, weight_decay=1e-05):
    """
    GCN model experiment run multiple times with accuracy and F1 score as output.

    Parameters:
    - graph_data: PyTorch Geometric Data object with x, edge_index, and y attributes
    - input_features: Number of input features for the model
    - num_classes: Number of classes in the output layer
    - n_experiments: Number of experiments to run
    - epochs: Number of training epochs
    - lr: Learning rate
    - weight_decay: Weight decay for the optimizer
    """
    
    class GCN(torch.nn.Module):
        def __init__(self, input_features, num_classes=4, hidden_dim=64):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(input_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv3(x, edge_index)
            return x


    # Run multiple experiments
    accuracies = []
    f1_scores = []
    
    for i in range(n_experiments):
        # Initialize model, optimizer, and loss function
        model = GCN(input_features, num_classes)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(graph_data)
            loss = criterion(out, graph_data.y)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(graph_data)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == graph_data.y).sum().item() / len(graph_data.y)
            f1 = f1_score(graph_data.y.cpu(), predictions.cpu(), average='macro')
        
        # Store results
        accuracies.append(accuracy)
        f1_scores.append(f1)
        print(f'Experiment {i+1}, Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1:.2f}')

    # Print average results
    print(f'\nAverage Accuracy over {n_experiments} experiments: {np.mean(accuracies) * 100:.2f}%')
    print(f'Average F1 Score over {n_experiments} experiments: {np.mean(f1_scores):.2f}')


# 9. main function
if __name__ == '__main__':
    X,y = load_data('C:/Users/Admin/Desktop/NN_assignment3/data/abalone.csv')

    # Decision Tree
    print("Running Decision Tree Experiments")
    decision_tree_experiment_multiple(X, y, n_experiments=10)

    # Random Forest
    print("\nRunning Random Forest Experiments")
    random_forest_experiment_multiple(X, y, n_experiments=10)

    # Simple Neural Network 
    print("\nRunning Simple Neural Network Experiments")
    neural_network_experiment_multiple(X, y, n_experiments=10)

    #GNN
    print("\nRunning Graph Neural Network Experiments")
    graph_data, input_features = create_graph_data(X, y)
    gcn_experiment_multiple(graph_data, input_features , n_experiments=10)