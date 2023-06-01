#!/usr/bin/env python
# coding: utf-8

# # EMAIL SPAM DETECTION WITH MACHINE LEARNING

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'C:/Users/lakav/Downloads/spam.csv', encoding='ISO-8859-1')
df.head()


# In[3]:


df.tail()


# In[4]:


df.describe()


# In[5]:


#show the no.of missing(NAN,NaN,na)data for each column
df.isnull().sum()


# In[6]:


#removing unneccessary columns from the dataset
df.drop(['Unnamed: 2',	'Unnamed: 3',	'Unnamed: 4'],axis=1,inplace=True)


# In[7]:


#changing the column name for better understanding(V1,V2 to SPAM,TEXT)
df.rename({'v1': 'SPAM','v2': 'TEXT'},axis=1,inplace=True)
df.head()


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(keep='first',inplace=True)


# In[10]:


df.duplicated().sum()


# In[11]:


#converting SPAM column to numerical values
df['SPAM'] = df['SPAM'].map({'ham': 0, 'spam': 1})


# VISUALIZATION

# In[12]:


plt.pie(df['SPAM'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# SPLITTING INTO TRAINING AND TESTING DATA

# In[13]:


X=df['TEXT']
y=df['SPAM']


# In[14]:


X


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=43)


# In[17]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[18]:


#preprocessing tool or utility class (LABELENCODER)
from sklearn.preprocessing import LabelEncoder
ec=LabelEncoder()
df['SPAM']=ec.fit_transform(df['SPAM'])
df['SPAM']


# In[19]:


ec=LabelEncoder()
y_train = ec.fit_transform(y_train)
y_test = ec.transform(y_test)


# In[20]:


print(X_train)
print(X_test)


# In[21]:


#feature extraction technique (COUNTVECTORIZER)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

# Fit the vectorizer on the training data and transform the training data
X_train_cv = cv.fit_transform(X_train)

# Transform the testing data using the fitted vectorizer
X_test_cv = cv.transform(X_test)
print(X_train_cv)


# IMPORTING MACHINE LEARNING MODELS

# In[22]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# 01-LOGISTIC REGRESSION

# In[23]:


#create and train the LOGISTIC REGRESSION
ln=LogisticRegression()
ln.fit(X_train_cv,y_train)
prediction_train=ln.predict(X_train_cv)
prediction_test=ln.predict(X_test_cv)


# In[24]:


#printing predictions and actual values after trained
print(ln.predict(X_train_cv))
print(y_train)


# In[25]:


#Evaluate the model on the training dataset
from sklearn.metrics import accuracy_score,confusion_matrix
print('confusion matrix:\n',confusion_matrix(y_train,prediction_train))
print('Accuracy:',accuracy_score(y_train,prediction_train))


# In[26]:


#printing predictions and actual values after testing
print(ln.predict(X_test_cv))
print(y_test)


# In[27]:


#Evaluate the model on the testing dataset
print('confusion matrix:\n',confusion_matrix(y_test,prediction_test))
print('Accuracy_testing:', accuracy_score(y_test,prediction_test))


# 02-Multinomial Naive Bayes

# In[28]:


#create and train the NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
naive_bayes=MultinomialNB().fit(X_train_cv,y_train)
naive_bayes.fit(X_train_cv,y_train)


# In[31]:


#printing predictions and actual values after trained
print(naive_bayes.predict(X_train_cv))
print(y_train)


# In[35]:


#Evaluate the model on the training dataset
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
prediction=naive_bayes.predict(X_train_cv)
print(classification_report(y_train,prediction))
print()
print('confusion Matrix:\n',confusion_matrix(y_train,prediction))
print()
print('Accuracy:', accuracy_score(y_train,prediction))


# In[33]:


#printing predictions and actual values after testing
print(naive_bayes.predict(X_test_cv))
print(y_test)


# In[34]:


#Evaluate the model on the testing dataset
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
pred=naive_bayes.predict(X_test_cv)
print(classification_report(y_test,pred))
print()
print('confusion Matrix:\n',confusion_matrix(y_test,pred))
print()
print('Accuracy:', accuracy_score(y_test,pred))


# COMPLETED
