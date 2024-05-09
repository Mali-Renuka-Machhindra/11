import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Get File from Drive using file-ID
import pandas as pd
!pip install PyDrive

data = pd.read_csv('Energy_consumption.csv')
data

# a) Find standard deviation, variance of every numerical attribute
std_deviation = data.std()
variance = data.var()
print("Standard Deviation:\n", std_deviation)
print("\nVariance:\n", variance)

# b) Find covariance and perform Correlation analysis.
covariance_matrix = data.cov()
correlation_matrix = data.corr()
print("\nCovariance Matrix:\n", covariance_matrix)
print("\nCorrelation Matrix:\n", correlation_matrix)

# c) How many independent features are present in the given dataset?
#Correlation finds the direction of the relationship along with the degree of the relationship.
independent_features = np.linalg.matrix_rank(data.corr())
print("\nNumber of independent features:", independent_features)

# d) Can we identify unwanted features?
# we can analyze correlation coefficients to identify features strongly correlated with each other.
# Features with high correlation might be candidates for removal if redundancy is present.
# Features with high correlation might be candidates for removal.
#Correlation shows us both, the direction and magnitude of how two quantities vary with each other.

unwanted_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            unwanted_features.add(colname)

print("\nd) Unwanted features:", unwanted_features)


# e) Perform data discretization using equi-frequency binning method on the age attribute
num_bins = 5  # Choose the number of bins as needed
data['age_bins'] = pd.qcut(data['Temperature'], q=num_bins, labels=False)


import matplotlib.pyplot as plt
# Visualize the equi-frequency bins graphically
plt.figure(figsize=(5, 3))
plt.hist(data['Temperature'], bins=num_bins, edgecolor='black', alpha=0.7)
plt.title('Equi-Frequency Binning for Age Attribute')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()


from sklearn import preprocessing
# f) Normalize RestBP, chol, and MaxHR attributes using different normalization techniques
attributes_to_normalize = ['RestBP', 'Chol', 'MaxHR']

# Min-Max normalization
min_max_scaler = preprocessing.MinMaxScaler()
data_min_max_normalized = data.copy()
print(set(data.columns) - set(attributes_to_normalize))

# Z-score normalization
z_score_scaler = StandardScaler()
data_z_score_normalized = data.copy()
attributes_to_normalize = [attribute for attribute in attributes_to_normalize if attribute in data.columns]

# Decimal scaling normalization
decimal_scaling_factor = 1000
data_decimal_scaled = data.copy()
data_decimal_scaled[attributes_to_normalize] = data[attributes_to_normalize] / decimal_scaling_factor

# Display the results
print("\nMin-Max Normalized DataFrame:\n", data_min_max_normalized.head())
print("\nZ-Score Normalized DataFrame:\n", data_z_score_normalized.head())
print("\nDecimal Scaled DataFrame:\n", data_decimal_scaled.head())

