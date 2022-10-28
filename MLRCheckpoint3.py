#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('/Users/brandonvidro/Downloads/project-iste-780-bad-project1/LassoRegression/LassoForwardSelectionDataSetForModelling.csv')


# In[3]:


df.head()


# In[6]:


X = df[['Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Cereal yield (kg per hectare)', 'Urban population (% of total)', 'GDP per capita (current US$)', 'Merchandise exports by the reporting economy, residual (% of total merchandise exports)', 'Permanent cropland (% of land area)', 'Merchandise imports by the reporting economy, residual (% of total merchandise imports)', 'Population density (people per sq. km of land area)', 'Agricultural land (% of land area)', 'Arable land (hectares per person)', 'Arable land (% of land area)', 'CO2 emissions from solid fuel consumption (kt)', 'Merchandise trade (% of GDP)']]


# In[7]:


y = df['LE']


# In[8]:


from sklearn import linear_model


# In[9]:


regr = linear_model.LinearRegression()


# In[13]:


regr.fit(X, y)


# In[14]:


print('Intercept \n', regr.intercept_)


# In[15]:


print('Coefficients: \n', regr.coef_)


# In[16]:


import statsmodels.api as sm


# In[17]:


X = sm.add_constant(X)


# In[18]:


model = sm.OLS(y, X).fit()


# In[19]:


predictions = model.predict(X)


# In[20]:


print_model = model.summary()


# In[21]:


print(print_model)


# In[ ]:




