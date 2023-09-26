#!/usr/bin/env python
# coding: utf-8

# # WINE QUALITY PREDICTION :

# # Importing the Dependencies

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# # Data Collection

# In[46]:


# loading the dataset to a Pandas DataFrame
wine_dataset = pd.read_csv(r'C:\Users\hhars\Desktop\Wine Quality Prediction\Wine Quality Data\winequality-red.csv')


# In[47]:


# number of rows & columns in the dataset
wine_dataset.shape


# In[48]:


# first 5 rows of the dataset
wine_dataset.head()


# In[49]:


# checking for missing values
wine_dataset.isnull().sum()


# # Data Analysis and Visulaization

# In[50]:


# statistical measures of the dataset
wine_dataset.describe()


# In[51]:


# number of values for each quality
sns.catplot(x='quality', data = wine_dataset, kind = 'count')


# In[52]:


# volatile acidity vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'volatile acidity', data = wine_dataset)


# In[53]:


# citric acid vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'citric acid', data = wine_dataset)


# In[54]:


# residual sugar vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'residual sugar', data = wine_dataset)


# In[55]:


# chlorides vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'chlorides', data = wine_dataset)


# In[56]:


# free sulfur dioxide vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'free sulfur dioxide', data = wine_dataset)


# In[57]:


# total sulfur dioxide vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'total sulfur dioxide', data = wine_dataset)


# In[58]:


# density vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'density', data = wine_dataset)


# In[59]:


# pH vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'pH', data = wine_dataset)


# In[60]:


# sulphates vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'sulphates', data = wine_dataset)


# In[61]:


# alcohol vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'alcohol', data = wine_dataset)


# # Correlation
# 
# * Positive Correlation
# * Negative Correlation

# In[62]:


correlation = wine_dataset.corr()


# In[80]:


# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')


# # Data Preprocessing

# In[64]:


# separate the data and Label
X = wine_dataset.drop('quality',axis=1)


# In[65]:


print(X)


# # Label Binarizaton

# In[66]:


Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[67]:


print(Y)


# # Train & Test Split

# In[68]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[69]:


print(Y.shape, Y_train.shape, Y_test.shape)


# # Model Training:
# 
# * Random Forest Classifier

# In[70]:


model = RandomForestClassifier()


# In[71]:


model.fit(X_train, Y_train)


# # Model Evaluation
# 
# * Accuracy Score

# In[72]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[73]:


print('Accuracy : ', test_data_accuracy)


# # Building a Predictive System

# In[74]:


input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')


# In[75]:


input_data = (7.9, 0.35, 0.46, 3.6, 0.078, 15.0, 37.0, 0.9973, 3.35, 0.86, 12.8)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')

