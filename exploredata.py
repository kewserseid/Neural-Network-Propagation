#It is better If this code snippet is executed in google collab 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load Wine dataset
wine = datasets.load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# Display basic dataset information
print("Dataset Overview:")
print(df.head())  # Display first few rows
print("\nSummary Statistics:")
print(df.describe())  # Summary statistics

# Class distribution
print("\nClass Distribution:")
print(df["target"].value_counts())

# Plot 1: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Plot 2: Boxplot of Alcohol Content Across Classes
plt.figure(figsize=(8, 6))
sns.boxplot(x="target", y="alcohol", data=df, palette="Set2")
plt.xlabel("Wine Class")
plt.ylabel("Alcohol Content")
plt.title("Distribution of Alcohol Content Across Classes")
plt.savefig("alcohol_distribution.png")
plt.show()
