import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# Fetch dataset
contraceptive_method_choice = fetch_ucirepo(id=30)

# Load data as pandas DataFrames
X = contraceptive_method_choice.data.features
y = contraceptive_method_choice.data.targets

# Convert `y` to a Series with the name 'Contraceptive_method' to avoid renaming issues
y = pd.Series(y.values.ravel(), name='Contraceptive_method')

# Combine features and target into a single DataFrame
data = pd.concat([X, y], axis=1)

# Step 1: Encode the 'Contraceptive_method' target variable
print("Unique categories in 'Contraceptive_method':", data['Contraceptive_method'].unique())
label_encoder = LabelEncoder()
data['Contraceptive_method'] = label_encoder.fit_transform(data['Contraceptive_method'])
print(data['Contraceptive_method'].head())

# Step 2: One-hot encode categorical features
print("Unique values in 'wife_religion':", data['wife_religion'].unique())
print("Unique values in 'wife_working':", data['wife_working'].unique())
print("Unique values in 'husband_occupation':", data['husband_occupation'].unique())
data = pd.get_dummies(data, columns=['wife_religion', 'wife_working', 'husband_occupation'], drop_first=True)
print(data.head())

# Step 3: Save the processed DataFrame to a CSV file
data.to_csv('C:/Users/Admin/Desktop/NN_assignment3/data/contraceptive_method_choice.csv', index=False)

