# Neural Network Assignment 3

This project focuses on analyzing and comparing various machine learning and deep learning models for multiclass classification tasks in biological and healthcare datasets. Specifically, the **Abalone Age Prediction** and **Contraceptive Method Choice** datasets are used to evaluate performance across different models and optimizers.

---

## Overview

The project implements and evaluates the following models:
1. **Decision Trees**
2. **Random Forests**
3. **Gradient Boosting/XGBoost**
4. **Neural Networks (NN)**
5. **Graph Neural Networks (GNN)**

### Goals:
- Compare the performance of models on structured datasets.
- Explore the impact of optimizers (Adam vs. SGD) on Neural Network training.
- Test the transferability of models across domains using the Contraceptive Method Choice dataset.
- Visualize and interpret model outputs to gain insights into feature importance.

---

## Project Structure

```plaintext
NN_ASSIGNMENT3/
│
├── data/                           # Contains datasets and data preparation scripts
│   ├── abalone_data_prepare.py     # Script for preparing the Abalone dataset
│   ├── abalone.csv                 # Main Abalone dataset
│   ├── abalone_infant.csv          # Subset of the Abalone dataset
│   ├── contraceptive_method_choice.csv # Healthcare dataset
│   └── data_prepare_cmc.py         # Script for preparing the healthcare dataset
│
├── notebooks/                      # Jupyter notebooks for experiments(result parts)
│   ├── data_analysis.ipynb         # Exploratory data analysis (EDA)
│   ├── graph_neural_network.ipynb  # Exploratory research on GNN
│   ├── RQ2_decision_tree.ipynb     # Decision tree experiments
│   ├── RQ3_decision_tree.ipynb     # Hyperparameter tuning for decision trees
│   ├── RQ4_random_forest.ipynb     # Random forest experiments
│   ├── RQ5_boost.ipynb             # Gradient Boosting (XGBoost) experiments
│   ├── RQ6_adam_sgd.ipynb          # Comparison of Adam vs. SGD optimizers
│   ├── RQ7_adam_hyper.ipynb        # Adam optimizer hyperparameter tuning
│   └── RQ8_transferability.ipynb   # Testing cross-domain transferability(Part B)
│
├── src/                            # Source code for custom models and utilities
│   ├── model_gnn.py                # Graph Neural Network implementation
│   ├── model_parta.py              # Code for abalone dataset models (Part A)
│   └── model_partb.py              # Code for healthcare dataset models (Part B)
│
├── requirements.txt                # requirements
├── .gitignore                      # Git ignore file
└── README.md                       # Project README file
```
---

## Our work for specific problem

### Part A

1. Analyse and visualise the given data sets by reporting the distribution of class, distribution of features and any other visualisation you find appropriate. **[II-A Data processing]**

2. Create a Decision Tree for the Abalone multi-class data and report train and test performance for multiple experimental (**can be 5 or more**) runs using different hyperparameters - i.e., **tree depth or any other hyperparameter of your choice**. Take the best Tree and report the Tree Visualisation (show your tree and also translate few selected nodes and leaves into IF and Then rules): **Note: Since Decision Trees give the same results for the same dataset, ensure that in different experimental runs, you create different set of train/test split as done in Week1 and Week 2 Exercise solutions. **[II-B Modeling][III-B result]**

   Let us review the method we used in Week 1 and Week 2 Exercise.
   
   ```python
   def read_data(run_num):
       data_in = genfromtxt("pima-indians-diabetes.csv", delimiter=",")
       data_inputx = data_in[:,0:8]  
       data_inputy = data_in[:,-1]   
   
       # train_test_split 
       x_train, x_test, y_train, y_test = train_test_split(
           data_inputx, data_inputy, test_size=0.40, random_state=run_num
       )
   
       return x_train, x_test, y_train, y_test
    
3. Do an investigation about improving performance further by either pre-pruning or post-pruning the tree: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html. **[II-B Modeling][III-B result]**
4. Apply Random Forests and show performance (eg. accuracy score) as your number of trees in the ensembles increases. **[II-C Random Forest Modeling][III-C result]**
5. Further, compare your results with XGBoost and Gradient Boosting and provide a discussion. **[II-D Boosting Comparison][III-D Discussion]**
6. Compare results with **Adam/SGD (Simple Neural Networks)** and discuss them. You can use default hyper-parameters from the sklearn library - there is no need for extensive hyperparameter search. **[II-E Neural Network Comparison][III-E Discussion]**
7. Using Adam, compare L2 regularisation (weight decay) with dropouts. Show results for 3 different combinations **(can be more)** of hyperparameters (dropout rate with weight decay hyper-parameter (λ) ).**[II-F Regularization Comparison][III-F result]**

### Part B

1. Provide data visualisation and then apply two of the best models from the above steps to the following dataset. You can report the results with the most appropriate metrics, i.e., F1, ROC-AUC, etc.**[IV-A Data Visualisation][IV-B Model Application]**
