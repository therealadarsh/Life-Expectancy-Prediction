#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import utils


# In[2]:


# Load dataset
#dataset = pd.read_csv('../featureSelectedDataset.csv')
dataset = pd.read_csv('../../LassoRegression/LassoForwardSelectionDataSetForModelling.csv')  
dataset.head()


# In[3]:


# Split the dataset into features and obs
forwardSelectionFeatures = pd.read_csv('../../LassoRegression/ForwardSelectionVia5FoldCV.csv')["0"].tolist()
#xLasso = df[np.intersect1d(df.columns, lassoFeatures)]
X = dataset[np.intersect1d(dataset.columns, forwardSelectionFeatures)]
#X = dataset.iloc[:,0:10]
y = dataset["Life expectancy at birth, total (years)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[4]:


X_train.head()


# In[5]:


y_train.head()


# In[6]:


kreg = KNeighborsRegressor(n_neighbors = 3)
kreg.fit(X_train.values, y_train.values)


# In[7]:


y_pred = kreg.predict(X_test.values)


# In[8]:


print('Accuracy training : {:.3f}'.format(kreg.score(X_train.values, y_train.values)))
print('Accuracy testing : {:.3f}'.format(kreg.score(X_test.values, y_test.values)))


# In[9]:


residuals = y_pred-y_test
residuals
residuals.shape, X_test.shape


# In[10]:


import matplotlib.pyplot as plt
import numpy as np
import os


# In[11]:


X_test.iloc[:,0]


# In[12]:


X_test.columns[0]


# In[13]:


def residual(x,y,title,folder):
    isExist = os.path.exists(folder)

    if not isExist:

      # Create a new directory because it does not exist 
      os.makedirs(folder)
    f = plt.figure()
    f = plt.scatter(x, y)
    plt.title(title)
    plt.savefig(folder + '/' + title + '.png', bbox_inches='tight', dpi=300)


# In[14]:


for index in range(0,10):
    residual(X_test.iloc[:,index], residuals,X_test.columns[index],'KNNRegression')


# In[ ]:




