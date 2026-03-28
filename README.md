# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the energy dataset.



2.Preprocess the data by handling missing values and standardizing the features.



3.Apply PCA to reduce the dimensionality of the dataset by selecting principal components.



4.Transform the data into reduced dimensions and analyze the variance explained by each component
 

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('HeightsWeights.csv')
print("First 5 rows of the dataset:")
print(data.head())
X=data[['Height(Inches)','Weight(Pounds)']]
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)',y='Weight(Pounds)',data=data)
plt.title('Original Data Distribution')
plt.show()

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
print("Explained Variance Ratio: ",pca.explained_variance_ratio_)
pca_df=pd.DataFrame(X_pca,columns=['PC1','PC2'])

plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1',y='PC2',data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:
<img width="712" height="617" alt="image" src="https://github.com/user-attachments/assets/1a5888dd-52f8-436e-949a-f64426d53a59" />


![WhatsApp Image 2026-03-28 at 3 48 39 PM](https://github.com/user-attachments/assets/3e58ff30-85e9-4b11-970a-baecee442186)




## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
