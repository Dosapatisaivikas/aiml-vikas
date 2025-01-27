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


# In[11]:


#changiing column names
data1.rename({'Solar.R':'Solar'},axis = 1,inplace = True)
data1.rename({'Temp':'Temperature'},axis = 1,inplace = True)
data1


# In[12]:


data1.info()


# In[13]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[14]:


#Visuvalize the data1 missing values usng graph
cols = data1.columns
colours = ['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[15]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[16]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of solar: ",median_solar)
print("Mean of solar: ",mean_solar)


# In[18]:


data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()


# In[41]:


print(data1['Weather'].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)



# In[43]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[45]:


print(data1['Month'].value_counts())
mode_Month = data1["Month"].mode()[0]
print(mode_Month)


# In[47]:


data1["Month"] = data1["Month"].fillna(mode_Month)
data1.isnull().sum()


# In[49]:


data1.tail()


# In[51]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[53]:


data1.tail()


# In[55]:


data1.reset_index(drop=True)


# In[63]:


fig,axes = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [ 1,3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0],color='skyblue',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# ### Observations
# * The Ozone column has extreme values beyound 81 as seen from boxplot
# * The same is comfigured from the below right-sweked histogram

# In[66]:


fig,axes = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [ 1,3]})
sns.boxplot(data=data1["Solar"], ax=axes[0],color='skyblue',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")
sns.histplot(data1["Solar"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# ### Observations
# * Solar has no outliers
# * Distribution is not exactly symmetric but slightly left sweked 

# In[ ]:




