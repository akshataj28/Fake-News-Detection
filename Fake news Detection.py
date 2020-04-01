#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# In[6]:

df=pd.read_csv("news.csv")
df.shape
df.head()


# In[7]:


labels=df.label
labels.head()


# In[8]:


#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[9]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[12]:


#Initialize a PassiveAggressiveClassifier
p=PassiveAggressiveClassifier(max_iter=50)
p.fit(tfidf_train,y_train)
#Predict on the test set and calculate accuracy
predict=p.predict(tfidf_test)
print(predict)
score=accuracy_score(y_test,predict)
accuracy=score*100
print('Accuracy: ',accuracy,'%')


# In[11]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




