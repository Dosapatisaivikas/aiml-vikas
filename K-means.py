#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[7]:


univ = pd.read_csv("Universities.csv")
univ


# In[9]:


univ.info()


# In[17]:


univ.isnull()


# In[21]:


plt.hist(univ["SAT"])
plt.show()


# In[31]:


univ.head(11)


# In[11]:


univ.describe()


# In[39]:


univ1 = univ.iloc[:,1:]


# In[41]:


univ1


# In[54]:


cols = univ1.columns


# In[58]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_univ_df = pd.DataFrame(scaler.fit_transform(univ1),columns = cols)
scaled_univ_df


# In[ ]:





# In[ ]:




