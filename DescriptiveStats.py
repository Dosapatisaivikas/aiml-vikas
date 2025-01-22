#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[7]:


df = pd.read_csv("Universities.csv")
df


# In[5]:


np.mean(df["SAT"])


# In[7]:


np.median(df["SAT"])


# In[11]:


np.mean(df["GradRate"])


# In[13]:


np.median(df["GradRate"])


# In[15]:


np.var(df["SAT"])


# In[17]:


df.describe()


# VISUVALIZATION

# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


plt.figure(figsize=(6,3))
plt.title("Acceptance ratio")
plt.hist(df["Accept"])


# In[52]:


sns.histplot(df["Accept"], kde = True)


# In[54]:


plt.figure(figsize=(6,3))
plt.title("GradRate ratio")
plt.hist(df["GradRate"])


# In[56]:


sns.histplot(df["GradRate"], kde = True)


# In[58]:


sns.histplot(df["SAT"], kde = True)


# ### OBSERVATIONS
#  * Acceptance ratio is non symmentric it is a right skewed
#  * GradRate ratio is non symmentric it is a left skewed
#  * SAT ratio is also a non symmentric it is a left skewed 

# In[ ]:




