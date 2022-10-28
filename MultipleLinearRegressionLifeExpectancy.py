#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('/Users/brandonvidro/Desktop/featureSelectedDataset.csv')


# In[5]:


df.head()


# In[14]:


X = df[['Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Arable land (% of land area)', 'Arable land (hectares per person)', 'Birth rate, crude (per 1,000 people)', 'CO2 emissions from solid fuel consumption (% of total)', 'Crop production index (2004-2006 = 100)', 'Livestock production index (2004-2006 = 100)', 'Permanent cropland (% of land area)', 'Population, female (% of total)', 'Rural population (% of total population)']]


# In[15]:


y = df['Life expectancy at birth, total (years)']


# In[16]:


from sklearn import linear_model


# In[17]:


regr = linear_model.LinearRegression()


# In[18]:


regr.fit(X, y)


# In[21]:


print('Intercept \n', regr.intercept_)


# In[22]:


print('Coefficients: \n', regr.coef_)


# In[23]:


import statsmodels.api as sm


# In[24]:


X = sm.add_constant(X)


# In[26]:


model = sm.OLS(y, X).fit()


# In[27]:


predictions = model.predict(X)


# In[28]:


print_model = model.summary()


# In[29]:


print(print_model)


# In[ ]:




