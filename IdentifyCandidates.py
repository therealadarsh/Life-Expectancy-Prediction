#!/usr/bin/env python
# coding: utf-8

# ### Indetify Candidates
# This code filters all the indicators to those having at least 10000 and present the results in a stacked format
# 
# 2021-09-14: DXG
# First pass filtering data down to 60 potential components that have over 10000 observations per predicted variable. Stacked version of data is in dataSetForModelling.csv
# 
# 2021-09-21: DXG
# Modified to pivot stacked data into wide format with no na's
# finalDataSetForModelling has the following columns:
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


#cleanedDataSetForModelling


# In[49]:


finalDataSetForModelling = cleanedDataSetForModelling.pivot(index=['CountryName','Year','Life expectancy at birth, total (years)'], columns='IndicatorName', values='Value')


# In[50]:


finalDataSetForModelling = finalDataSetForModelling.reset_index().dropna()


# In[51]:


finalDataSetForModelling = finalDataSetForModelling.reset_index().drop(['index'], axis=1)


# In[53]:


finalDataSetForModelling.shape
# CountryName,Year,Life expectancy at birth, total (years), +60 predictors


# ### Add column for overall average and average per year

# In[62]:


meanByYear = finalDataSetForModelling[['Year','Life expectancy at birth, total (years)']].groupby('Year').mean().reset_index()
meanOverall = finalDataSetForModelling[['Life expectancy at birth, total (years)']].mean()

finalDataSetForModelling['MeanLifeExpetancyOverall'] = meanOverall[0]

meanByYear= meanByYear.rename(columns={'Life expectancy at birth, total (years)':'MeanLifeExpetancyForYear'})

finalDataSetForModelling = pd.merge(left=finalDataSetForModelling,right=meanByYear)

finalDataSetForModelling['AboveAverageLifeExpectancyOverall'] = finalDataSetForModelling['Life expectancy at birth, total (years)']>finalDataSetForModelling['MeanLifeExpetancyOverall']

finalDataSetForModelling['AboveAverageLifeExpectancyByYear'] = finalDataSetForModelling['Life expectancy at birth, total (years)']>finalDataSetForModelling['MeanLifeExpetancyForYear']

finalDataSetForModelling.to_csv("finalDataSetForModelling.csv", index=False)


# # Everything below this is just exploratory code
# Trying out different modelling scenarios

# In[65]:


components = dataSetForModelling['IndicatorName'].unique()


# In[22]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[66]:


components[0]


# In[23]:


model = LinearRegression()


# In[82]:


filteredDataset['Value'].values.reshape((-1, 1))


# In[97]:


filteredDataset[4:5] #.values.reshape((-1, 1))


# In[83]:


filteredDataset=dataSetForModelling[dataSetForModelling['IndicatorName']==components[0]]
x=filteredDataset['Value'].values.reshape((-1, 1))
y=filteredDataset[lifeExpectancyTotal]
model.fit(x,y)


# In[84]:


model.score(x,y)


# In[77]:


y


# # Great stuff
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html

# In[85]:


import numpy as np
import statsmodels.api as sm


# In[86]:


x = sm.add_constant(x)


# In[87]:


x


# In[88]:


model = sm.OLS(y, x)


# In[89]:


results = model.fit()


# In[90]:


results.summary()


# In[93]:


results.pvalues[1]


# In[94]:


results.tvalues


# In[95]:


results.rsquared_adj


# In[ ]:




