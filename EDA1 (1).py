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


# In[6]:


# droping columns
data1 = data.drop(['Unnamed: 0','Temp C'],axis = 1)
data1


# In[7]:


data1.info()


# In[8]:


# converting month data type from object to integer type
data1['Month']=pd.to_numeric(data['Month'],errors = 'coerce')
data1.info()


# In[9]:


# checking for duplicate rows
data1[data1.duplicated(keep = False)]


# In[10]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[28]:


#changiing column names
data1.rename({'Solar.R':'Solar'},axis = 1,inplace = True)
data1.rename({'Temp':'Temperature'},axis = 1,inplace = True)
data1


# In[30]:


data1.info()


# In[32]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[47]:


#Visuvalize the data1 missing values usng graph
cols = data1.columns
colours = ['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[55]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[53]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[57]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of solar: ",median_solar)
print("Mean of solar: ",mean_solar)


# In[59]:


data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()


# In[ ]:






# In[ ]:




