#!/usr/bin/env python
# coding: utf-8

# ## This is a template to use for training classification and regression models
# We have two feature sets that we need to decide on which ones to use for our modelling.
# 
# Use linear regression to determine which set performs better in prediction accuracy so that we can make suggest action to be taken based on inference
# 
# Then determine the best classification n for knn to determine the best model for predicting an above average outcome for life expectancy for the feature set decided upon with the linear regression

# In[9]:


import pandas as pd

df = pd.read_csv('LassoForwardSelectionDataSetForModelling.csv')
#lassoFeatures = pd.read_csv('top10predictorsLasso_cv.csv')['0'].tolist()
forwardSelectionFeatures = pd.read_csv('ForwardSelectionVia5FoldCV.csv')["0"].tolist()

y_qualitative = df['AboveAverageLifeExpectancyByYear']
y_quantitative = df['Life expectancy at birth, total (years)']
#xLasso = df[np.intersect1d(df.columns, lassoFeatures)]
xFowardSelection = df[np.intersect1d(df.columns, forwardSelectionFeatures)]

forwardSelectionFeatures


# In[ ]:




