#!/usr/bin/env python
# coding: utf-8
### Neural Network Timeseries classification 
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

DXG - 2021-10-02
# In[1]:


import plotly.graph_objs as go
from plotly.offline import iplot

def plot_dataset(df, title):
    data = []
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)


# In[11]:


# User defined function for accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


# In[12]:


# Load dataset
dataset = pd.read_csv('../featureSelectedDataset.csv')  
dataset.head()


# In[13]:


# Split the dataset into features and obs

X = dataset.iloc[:,0:10]
y = dataset["AboveAverageLifeExpectancyByYear"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[14]:


# Fit the model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)


# In[15]:


# Fit the model and get accuracy
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)

y_score = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print('Accuracy of MLPClassifier : ', accuracy(cm))


# In[24]:


y_pred_train = clf.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
print('Accuracy of MLPClassifier (training): ', accuracy(cm))


# In[23]:


# Generate ROC plot
fpr2, tpr2, threshold = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
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
plt.savefig('MLClassfierROC.jpg', dpi=300)
plt.show()


# In[ ]:




