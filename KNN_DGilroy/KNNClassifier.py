#!/usr/bin/env python
# coding: utf-8

# ### Multi Layer Perceptron Classifier 
# This will use the mulit layer perceptron classifier to construct a model on a training set, then perform analysis on the test data for performance evaluation.
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# 
# 
# DXG - 2021-09-25

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix


# In[2]:


# User defined function for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[3]:


# Load dataset
dataset = pd.read_csv('../featureSelectedDataset.csv')  
dataset.head()


# In[4]:


# Split the dataset into features and obs

X = dataset.iloc[:,0:10]
y = dataset["AboveAverageLifeExpectancyByYear"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[5]:


# Fit the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)


# In[6]:


y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print('Accuracy of MLPClassifier : ', accuracy(cm))


# In[7]:


y_pred_train = model.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
print('Accuracy of KNNClassifier (training): ', accuracy(cm))


# In[8]:


# Generate ROC plot
fpr2, tpr2, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc2 = auc(fpr2, tpr2)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
plt.figure()
lw = 2
plt.plot(fpr2, tpr2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('KNNClassfierROC.jpg', dpi=300)
plt.show()


# In[9]:


X


# In[ ]:




