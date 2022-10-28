#!/usr/bin/env python
# coding: utf-8

# ## Models two feature sets via linear regression
# We have two feature sets that we need to decide on which ones to use for our modelling.
# 
# Use linear regression to determine which set performs better in prediction accuracy so that we can make suggest action to be taken based on inference

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('LassoForwardSelectionDataSetForModelling.csv')
lassoFeatures = pd.read_csv('top10predictorsLasso_cv.csv')['0'].tolist()
forwardSelectionFeatures = pd.read_csv('ForwardSelectionVia5FoldCV.csv')["0"].tolist()


# In[3]:


y_qualitative = df['AboveAverageLifeExpectancyByYear']
y_quantitative = df["Life expectancy at birth, total (years)"]


# In[4]:


xLasso = df[np.intersect1d(df.columns, lassoFeatures)]
xFowardSelection = df[np.intersect1d(df.columns, forwardSelectionFeatures)]


# In[5]:


# Perform cross validation for Lasso features 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, xLasso, y_quantitative, cv=5)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[6]:


# Perform cross validation for Forward Selection features 
from sklearn.model_selection import cross_val_score
# Fit the model
model = LinearRegression()
cv_scores = cross_val_score(model, xFowardSelection, y_quantitative, cv=5)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# In[7]:


lassoFeatures


# In[8]:


forwardSelectionFeatures


# In[9]:


from sklearn import metrics
modelAll = LinearRegression()
modelAll.fit(xLasso, y_quantitative)
y_pred = modelAll.predict(xLasso)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_quantitative, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_quantitative, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_quantitative, y_pred)))
print('R-Squared:', modelAll.score(xLasso, y_quantitative))
print('VIF must be less than:', 1/(1-modelAll.score(xLasso, y_quantitative)))


# In[10]:


# Multicollinearity checks via VIF (An acceptable VIF is if it’s less than the max of 10 and 1/1-R² model)
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[11]:


X_variables = xLasso
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
vif_data.to_csv("vif_data_lasso.csv")
vif_data


# In[12]:


from sklearn import metrics
modelAll = LinearRegression()
modelfit = modelAll.fit(xFowardSelection, y_quantitative)
y_pred = modelAll.predict(xFowardSelection)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_quantitative, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_quantitative, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_quantitative, y_pred)))
print('R-Squared:', modelAll.score(xFowardSelection, y_quantitative))
print('VIF must be less than:', 1/(1-modelAll.score(xFowardSelection, y_quantitative)))


# In[13]:


X_variables = xFowardSelection
vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
vif_data.to_csv("vif_data_forward_selection.csv")
vif_data


# In[14]:


import statsmodels.api as sm
est = sm.OLS(y_quantitative,xFowardSelection)
est2 = est.fit()
print(est2.summary())


# In[ ]:




