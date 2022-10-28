#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[20]:


df = pd.read_csv('featureSelectedDataset.csv')


# In[21]:


g = sns.pairplot(df.iloc[:,1:11])

for ax in g.axes.flatten():
    # rotate x axis labels
    ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    # rotate y axis labels
    ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    # set y labels alignment
    ax.yaxis.get_label().set_horizontalalignment('right')


# In[30]:


figure = g.fig
figure.savefig('PairPlotsPredictorsAndY.png',bbox_inches='tight')


# In[ ]:




