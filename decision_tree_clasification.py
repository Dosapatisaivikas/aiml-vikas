#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[3]:


iris = pd.read_csv("iris.csv")


# In[5]:


iris


# In[7]:


iris.info()


# In[9]:


iris[iris.duplicated(keep=False)]


#  ### OBSERAVTIONS
#  * There are 150 rows and 5 cols
#  * There are one duplicated row
#  * There are  no missing values
#  * The x-columns are sepal.length,sepal.width,petal.length,sepal.width
#  * All the x-columns are contiounes
#  * The y-columns is varity which is categorical
#  * There are thre flower categries(class)

# In[27]:


iris = iris.drop_duplicates(keep = 'first')


# In[29]:


iris = iris.reset_index(drop=True)
iris


# In[ ]:




