# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
