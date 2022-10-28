#!/usr/bin/env python
# coding: utf-8

# ### Indentify Candidates
# This code filters all the indicators to those having at least 10000 and present the results in a stacked format
# 
# 2021-09-14: DXG
# First pass filtering data down to 60 potential components that have over 10000 observations per predicted variable. Stacked version of data is in dataSetForModelling.csv
# 
# 2021-09-21: DXG
# Modified to pivot stacked data into wide format with no na's
# finalDataSetForModelling.csv has the following columns:
# CountryName,Year,Life expectancy at birth, total (years), +60 predictors
# 

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('indicators.csv')


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df_groupBy = df.groupby('IndicatorName').size().reset_index(name='counts').sort_values('counts')


# In[6]:


df_groupBy.to_csv('IndicatorNameGroupedByCount.csv')


# In[7]:


#use fairly arbitrary cutoff at 10000 observations per indicator
cutOff = 10000
df_groupByOver10000 = df_groupBy[df_groupBy['counts'] > cutOff]


# In[8]:


df_groupByOver10000.to_csv('IndicatorNameGroupedByCountOver' + str(cutOff) + '.csv')


# In[9]:


#filter data to the cutoff limit
df_groupByOver10000.shape


# In[10]:


filteredDataset = pd.merge(left=df,right=df_groupByOver10000)


# In[11]:


filteredDataset.to_csv('filteredDataset.csv')


# In[12]:


lifeExpectancyTotal = 'Life expectancy at birth, total (years)'
lifeExpectancyMale = 'Life expectancy at birth, male (years)'
lifeExpectancyFemale = 'Life expectancy at birth, female (years)'
dependentVariables = [lifeExpectancyTotal,lifeExpectancyMale,lifeExpectancyFemale]


# In[13]:


dependentVariables


# In[14]:


#now need to split dataset into the indepedent and dependent sets
independentDataRaw = filteredDataset[filteredDataset.IndicatorName.isin(dependentVariables)]


# In[15]:


dependentData = filteredDataset[~filteredDataset.IndicatorName.isin(dependentVariables)]


# In[16]:


independentDataRawPivoted = pd.pivot(independentDataRaw,values='Value',index=['CountryName','CountryCode','Year'], columns='IndicatorName')


# In[17]:


independentDataRawPivotedFlattened = independentDataRawPivoted.reset_index(level=[0,1])


# In[18]:


independentDataRawPivotedFlattened.columns.to_flat_index()
independentData = pd.DataFrame(independentDataRawPivotedFlattened.to_records())


# In[19]:


independentData
dataSetForModelling =  pd.merge(left=dependentData,right=independentData)


# In[20]:


dataSetForModelling.to_csv('datasetForFeatureSelection.csv')


# In[21]:


cleanedDataSetForModelling = dataSetForModelling.drop(['IndicatorCode', 'counts'], axis=1)


# In[22]:


cleanedDataSetForModelling


# In[23]:


finalDataSetForModelling = cleanedDataSetForModelling.pivot(index=['CountryName','Year','Life expectancy at birth, total (years)'], columns='IndicatorName', values='Value')


# In[24]:


finalDataSetForModelling = finalDataSetForModelling.reset_index().dropna()


# In[25]:


finalDataSetForModelling = finalDataSetForModelling.reset_index().drop(['index'], axis=1)


# # finalDataSetForModelling.csv is the feature set of 60 features, each having at least 10000 observations
# This is used for further feature selection to get it down to 10 

# In[26]:


finalDataSetForModelling.to_csv("finalDataSetForModelling.csv", index=False)

