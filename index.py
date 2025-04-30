# Task 1: Load and explore Dataset
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
# Load Iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]  # Add species column

# Inspect the dataset
# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Task 2: Basic data analysis
print("Summary Statistics:")
print(df.describe())

species_mean = df.groupby('species').mean()
print("\nMean measurements per species:")
print(species_mean)

# Task 3: Data Visualization
# Line chart (Trend of sepal length by index)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title("Trend of Sepal Length Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# Bar chart (Average petal length by species)
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram (Distribution of sepal width)
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='green')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot (Sepal length vs Petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

