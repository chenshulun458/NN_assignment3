import pandas as pd
from ucimlrepo import fetch_ucirepo


# Fetch the Abalone dataset
abalone = fetch_ucirepo(id=1)

# Combine features and targets into a single DataFrame
data = pd.concat([abalone.data.features, abalone.data.targets], axis=1)

# Redefine the 'Sex' column to 'Infant' and 'Non-Infant'
data['Sex'] = data['Sex'].apply(lambda x: 'Infant' if x == 'I' else 'Non-Infant')
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'Non-Infant' else 0)

# Define the classification function for 'Rings'
def get_age_class(rings):
    if rings <= 7:
        return 1  # Class 1: 0 - 7 years
    elif rings <= 10:
        return 2  # Class 2: 8 - 10 years
    elif rings <= 15:
        return 3  # Class 3: 11 - 15 years
    else:
        return 4  # Class 4: Greater than 15 years

# Apply the function to classify 'Rings' and overwrite the column
data['Rings'] = data['Rings'].apply(get_age_class)

# Save the final processed data to 'data/abalone/abalone.csv'
data.to_csv('/data/abalone.csv', index=False)

print("Final processed data saved to 'abalone.csv'")


