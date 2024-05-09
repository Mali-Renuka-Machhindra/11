import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\Documents\\Energy_consumption.csv')
# print(data)

# a) Find Missing Values and replace with suitable alternatives
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# b) Remove inconsistency
# You need to define the inconsistency based on your dataset. For example, removing duplicates.
data.drop_duplicates(inplace=True)

# c) Boxplot analysis for each numerical attribute and find outliers
numerical_attributes = data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 5))
for col in numerical_attributes:
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# d) Draw histogram for any two suitable attributes
plt.figure(figsize=(12, 6))
plt.hist(data['Temperature'], bins=20, color='blue', alpha=0.7, label='Temperature')
plt.hist(data['Humidity'], bins=20, color='orange', alpha=0.7, label='Humidity')
plt.title("Histogram for Age and Chol Attributes")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# e) Find data type of each column
data_types = data.dtypes
print("Data Types:\n", data_types)

# f) Finding out Zeros
zeros_count = (data == 0).sum()
print("Zeros Count:\n", zeros_count)