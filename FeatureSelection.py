#!/usr/bin/env python
# coding: utf-8

# # Feature selection
# This code will work on various methods to identify the features having the most impact on the life expectancy. It is expected the year will have the larges impact, so will we will have at least 4 features to rank overall.
# 
# 2021-09-21 - DXG 

# In[30]:


import pandas as pd


# In[31]:


df = pd.read_csv('finalDataSetForModelling.csv')


# In[32]:


features = df.columns.tolist()


# In[33]:


features.remove('CountryName')
features.remove('Life expectancy at birth, total (years)')


# In[34]:


X=df[features]
y=df['Life expectancy at birth, total (years)']


# ## Feature selection is based on a crude forward selection via linear regression to find quantitative predictors having the greatest impact on the life expectancy
# The process is to find 10 at a time, then remove the ones that appear to obviously correlate with life expectancy, such as mortality rate, population growth, population ages as percentage of total.
# Continue the trimming down until we have 10 selected features that do not appear to have a direct correlation with life expectancy.
# Year is left out, as it is expected that Year would be directly correlated with life expectancy (and it's an uncontrollable predictor)

# In[37]:


#!pip install mlxtend


# In[38]:


# importing the models

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression


# In[39]:


def GetFeatures(k_features, X, y, verbose):
    # calling the linear regression model
    lreg = LinearRegression()
    sfs1 = sfs(lreg, k_features=k_features, forward=True, verbose=verbose, scoring='neg_mean_squared_error')

    sfs1 = sfs1.fit(X, y)

    feat_names = list(sfs1.k_feature_names_)
    return feat_names


# # First pass
# Obviously, we need to exclude features that are highly correlated with the life expectancy:
# So let's see what we have

# In[40]:


featurePass1 = GetFeatures(10,X,y,0)


# In[41]:


featurePass1


# #### Let's remove all the morality related to upper end of age and survival to certain ages

# In[42]:


featuresToRemove = [
 'Death rate, crude (per 1,000 people)',
 'Mortality rate, adult, female (per 1,000 female adults)',
 'Mortality rate, under-5 (per 1,000)',
 'Population ages 65 and above (% of total)',
 'Survival to age 65, female (% of cohort)',
 'Survival to age 65, male (% of cohort)']

X=X.drop(columns=featuresToRemove)


# In[43]:


featurePass2 = GetFeatures(10,X,y,0)


# In[44]:


featurePass2


# In[45]:


#### Let's remove all the morality related and Year, as well as relative population ages


# In[46]:


featuresToRemove2 = ['Year',
 'Mortality rate, adult, male (per 1,000 male adults)',
 'Mortality rate, infant (per 1,000 live births)',
 'Population, ages 15-64 (% of total)']
X=X.drop(columns=featuresToRemove2)


# In[47]:


featurePass3 = GetFeatures(10,X,y,0)


# In[48]:


featurePass3


# In[49]:


#### Remove age related indicators - fertilty rate is hard to control


# In[50]:


featuresToRemove3 = ['Age dependency ratio (% of working-age population)',
 'Age dependency ratio, old (% of working-age population)']
X=X.drop(columns=featuresToRemove3)


# In[51]:


featurePass4 = GetFeatures(10,X,y,0)
featurePass4


# #### remove age dependencies, as well as growth, but leave relative growth

# In[52]:


featuresToRemove4 = ['Age dependency ratio, young (% of working-age population)',
 'Population growth (annual %)',
 'Urban population growth (annual %)']
X=X.drop(columns=featuresToRemove4)


# In[53]:


featurePass5 = GetFeatures(10,X,y,0)
featurePass5


# In[54]:


featuresToRemove5 = [
 'Population, ages 0-14 (% of total)',
 'Rural population growth (annual %)']
X=X.drop(columns=featuresToRemove5)


# In[55]:


featurePass6 = GetFeatures(10,X,y,0)
featurePass6


# ### So we have 10 predictors that do not have an apparent direct correlation with life expectancy, so let's explore these.

# In[56]:


finalColumns = list(featurePass6)


# In[57]:


finalColumns.append('Life expectancy at birth, total (years)')
finalColumns.append('Year')
finalColumns.append('CountryName')


# In[58]:


featureSelectedDataset = df[finalColumns]
featureSelectedDataset.reset_index()


# In[59]:


meanByYear = featureSelectedDataset[['Year','Life expectancy at birth, total (years)']].groupby('Year').mean().reset_index()
meanOverall = featureSelectedDataset[['Life expectancy at birth, total (years)']].mean()

featureSelectedDataset['MeanLifeExpetancyOverall'] = meanOverall[0]

meanByYear= meanByYear.rename(columns={'Life expectancy at birth, total (years)':'MeanLifeExpetancyForYear'})

featureSelectedDataset = pd.merge(left=featureSelectedDataset,right=meanByYear)

featureSelectedDataset['AboveAverageLifeExpectancyOverall'] = featureSelectedDataset['Life expectancy at birth, total (years)']>featureSelectedDataset['MeanLifeExpetancyOverall']

featureSelectedDataset['AboveAverageLifeExpectancyByYear'] = featureSelectedDataset['Life expectancy at birth, total (years)']>featureSelectedDataset['MeanLifeExpetancyForYear']

featureSelectedDataset.to_csv("finalDataSetForModelling.csv", index=False)


# In[60]:


featureSelectedDataset


# In[61]:


featureSelectedDataset.to_csv('featureSelectedDataset.csv', index=False)


# In[62]:


print(*featureSelectedDataset.columns, sep = "\n")


# ### Final feature selected dataset has 10 predictors, with Year and Country added on for informational purposes
# Adolescent fertility rate (births per 1,000 women ages 15-19)
# Arable land (% of land area)
# Arable land (hectares per person)
# Birth rate, crude (per 1,000 people)
# CO2 emissions from solid fuel consumption (% of total)
# Crop production index (2004-2006 = 100)
# Livestock production index (2004-2006 = 100)
# Permanent cropland (% of land area)
# Population, female (% of total)
# Rural population (% of total population)
# Life expectancy at birth, total (years)
# Year
# CountryName
