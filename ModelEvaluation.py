#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


# In[3]:


#import the csv file using pandas.

#can edit file path and file name to match your machine
#data used is from https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning/code

file_path = 'C:/Users/jesmo/OneDrive/Desktop/'
file_name = 'Phishing_Legitimate_full.csv'

df = pd.read_csv(file_path + file_name)
df= df.drop(['id'], axis =1)


# In[4]:


X = df.drop(['CLASS_LABEL'], axis = 1)
y = df['CLASS_LABEL']


# In[6]:


def select_best_features(X, y):
    # compute mutual information scores between the variables, and order them based on the scores
    mi_scores = mutual_info_classif(X, y)
    top_features = X.columns[mi_scores.argsort()[::-1]]
    
    # create bar plot with labels rotated 90 degrees
    plt.figure(figsize=(10, 6))  # adjust width and height as desired
    plt.bar(range(len(top_features)), mi_scores[mi_scores.argsort()[::-1]])
    plt.xticks(range(len(top_features)), top_features, rotation=90, fontsize=12)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Mutual Information Score', fontsize=14)
    plt.title('Top Features by Mutual Information Score', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # return the top_features for X
    return X[top_features]

# save the results as variable X_top
X_top = select_best_features(X, y)

plt.scatter(X_top.iloc[:, 0], X_top.iloc[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Dataset')
plt.show()


# In[7]:


#instatiate the scaler model
scaler = StandardScaler()
pca = PCA()

pipeline = make_pipeline(scaler, pca)

pipeline.fit(X_top, y)

X_pt = pipeline.transform(X_top)

plt.scatter(X_pt[:, 0], X_pt[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot of first two principal components')
plt.colorbar()
plt.show()


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_pt, y, test_size=.25, random_state=42, stratify=y)


# In[10]:


models = {'Logistic Regression': LogisticRegression(max_iter = 10000),
         'KNN': KNeighborsClassifier(),
         'Decision Tree': DecisionTreeClassifier()}

results = []

for name, model in models.items():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf)
    results.append(cv_results)
    
plt.boxplot(results, labels=models.keys())
plt.show()


# In[11]:


for name, model in models.items():
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('{} Test: {}'.format(name, test_score))


# In[ ]:




