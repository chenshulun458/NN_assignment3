# Neural Network Assignment 3

This project focuses on analyzing and comparing various machine learning and deep learning models for multiclass classification tasks in biological and healthcare datasets. Specifically, the **Abalone Age Prediction** and **Contraceptive Method Choice** datasets are used to evaluate performance across different models and optimizers.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Work](#future-work)

---

## Overview

The project implements and evaluates the following models:
1. **Decision Trees**
2. **Random Forests**
3. **Gradient Boosting (XGBoost)**
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
├── notebooks/                      # Jupyter notebooks for model training and analysis
│   ├── data_analysis.ipynb         # Exploratory data analysis (EDA)
│   ├── graph_neural_network.ipynb  # Implementation and evaluation of GNNs
│   ├── RQ2_decision_tree.ipynb     # Decision tree experiments
│   ├── RQ3_decision_tree.ipynb     # Hyperparameter tuning for decision trees
│   ├── RQ4_random_forest.ipynb     # Random forest experiments
│   ├── RQ5_boost.ipynb             # Gradient Boosting (XGBoost) experiments
│   ├── RQ6_adam_sgd.ipynb          # Comparison of Adam vs. SGD optimizers
│   ├── RQ7_adam_hyper.ipynb        # Adam optimizer hyperparameter tuning
│   └── RQ8_transferability.ipynb   # Testing cross-domain transferability
│
├── src/                            # Source code for custom models and utilities
│   ├── model_gnn.py                # Graph Neural Network implementation
│   ├── model_parta.py              # Code for abalone dataset models
│   ├── model_partb.py              # Code for healthcare dataset models
│   └── output_graph/               # Directory for saving visualizations and plots
│
├── .gitignore                      # Git ignore file
└── README.md                       # Project README file
