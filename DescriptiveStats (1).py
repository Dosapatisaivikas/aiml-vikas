#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[3]:


np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[5]:


np.mean(df["GradRate"])


# In[6]:


np.median(df["GradRate"])


# In[7]:


np.var(df["SAT"])


# In[8]:


df.describe()


# VISUVALIZATION

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


plt.figure(figsize=(6,3))
plt.title("Acceptance ratio")
plt.hist(df["Accept"])


# In[12]:


sns.histplot(df["Accept"], kde = True)


# In[13]:


plt.figure(figsize=(6,3))
plt.title("GradRate ratio")
plt.hist(df["GradRate"])


# In[14]:


sns.histplot(df["GradRate"], kde = True)


# In[15]:


sns.histplot(df["SAT"], kde = True)


# ### OBSERVATIONS
#  * Acceptance ratio is non symmentric it is a right skewed
#  * GradRate ratio is non symmentric it is a left skewed
#  * SAT ratio is also a non symmentric it is a left skewed 

# In[32]:


s=[20,15,10,25,30,35,28,40,45,60]
scores1 = pd.Series(s)
scores1


# In[34]:


plt.boxplot(scores1,vert = False)


# In[50]:


s2 = [10,20,3,5,11,10,250]
scores2=pd.Series(s2)
scores2


# In[52]:


plt.boxplot(scores2,vert = False)


# In[86]:


df = pd.read_csv("Universities.csv")
plt.boxplot(df["SAT"],vert = False)


# In[88]:


plt.boxplot(df["Accept"],vert = False)


# In[90]:


plt.boxplot(df["GradRate"],vert = False)


# In[92]:


plt.boxplot(df["Expenses"],vert = False)


# In[94]:


plt.boxplot(df["SFRatio"],vert = False)


# In[96]:


plt.boxplot(df["Top10"],vert = False)


# ## obersvation on boxplot on university dataset
# * SAT has 4 outliers
# * accept has no outliers
# * Sf ratio has Outliers
# * expenses has outliers
# * AMONG ALL "SAT" HAS MORE OUTLIERS 

# In[ ]:




