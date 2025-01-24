#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)


# In[5]:


data.dtypes


# In[10]:


# droping columns
data1 = data.drop(['Unnamed: 0','Temp C'],axis = 1)
data1


# In[12]:


data1.info()


# In[23]:


# converting month data type from object to integer type
data1['Month']=pd.to_numeric(data['Month'],errors = 'coerce')
data1.info()


# In[27]:


# checking for duplicate rows
data1[data1.duplicated(keep = False)]


# In[ ]:




