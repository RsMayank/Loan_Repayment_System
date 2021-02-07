#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

print("import successful")


# In[24]:


data = pd.read_csv('dataLRS.csv',sep = ',',header = 0)
#Reading the data file


# In[25]:


print("DATA::")
data.head()


# In[26]:


data.shape


# In[70]:


X = data.values[:,0:4]
X[0]


# In[38]:


y = data.values[:,5]


# In[71]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)


# In[72]:


#dtc - decision tree clasifier (You can choose any variable name)
#this 'dtc' will pass to Decision Tree Clasifier 

dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
#max_depth : int, default=None
 #   The maximum depth of the tree. If None, then nodes are expanded until
  #  all leaves are pure or until all leaves contain less than
   # min_samples_split samples.
#min_samples_leaf : int or float, default=1
    #The minimum number of samples required to be at a leaf node.
    #A split point at any depth will only be considered if it leaves at
    #least ``min_samples_leaf`` training samples in each of the left and
    #right branches.  This may have the effect of smoothing the model,
    #especially in regression.
    
#If you are using jupyter notebook then use shift + TAB for help with function


# In[73]:


#Fitting the test data into the model
dtc.fit(X_train,y_train)


# In[74]:


#Now it is time to predict the Model accuracy

#using the test data which is saved in X_test and y_test

test_predct = dtc.predict(X_test)


# In[75]:


#Printing the output and accuracy score
test_predct


# In[76]:


#Accuracy Score

Model_score = accuracy_score(y_test,test_predct)*100


# In[77]:


Model_score


# In[ ]:


#Accuracy Score is 93.66 %

#twitter.com/@rs_mayank

