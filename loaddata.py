import pandas as pd
import os
import matplotlib.pyplot as plt
import urllib.request

#CSV

file_path = os.path.join("datasets", "housing", "housing.csv")
# os.makedirs(os.path.dirname(file_path), exist_ok=True)
# urllib.request.urlretrieve("", file_path)
housing = pd.read_csv(file_path)
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())
housing.hist()
housing.hist(bins=50, figsize=(20, 15))
plt.show()

#excel

file_path = os.path.join("datasets", "housing", "housing.xlsx")
housing = pd.read_excel(file_path)
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())
housing.hist()
housing.hist(bins=50, figsize=(20, 15))
plt.show()

