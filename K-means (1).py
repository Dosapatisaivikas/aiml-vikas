#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


univ = pd.read_csv("Universities.csv")
univ


# In[3]:


univ.info()


# In[4]:


univ.isnull()


# In[5]:


plt.hist(univ["SAT"])
plt.show()


# In[6]:


univ.head(11)


# In[7]:


univ.describe()


# In[8]:


univ1 = univ.iloc[:,1:]


# In[9]:


univ1


# In[10]:


cols = univ1.columns


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_univ_df = pd.DataFrame(scaler.fit_transform(univ1),columns = cols)
scaled_univ_df


# In[68]:


# building 3 clusters using kmeans cluster algorithm
from sklearn.cluster import KMeans
cluster_new = KMeans(3, random_state=0) # specifying 3 clusters
cluster_new.fit(scaled_univ_df)


# In[70]:


cluster_new.labels_
# print cluster labels


# In[72]:


set(cluster_new.labels_)


# In[74]:


univ['cluster_new'] = cluster_new.labels_


# In[76]:


univ


# In[78]:


univ.sort_values(by = 'cluster_new')


# In[80]:


univ.iloc[:,1:].groupby("cluster_new").mean()
# use groupby() to find aggreagated (mean) values in each cluster


# ### OBSERVATIONS 
# - cluster 2 as a top rated universities as the cut off score,top10,sfratio,parameters mean value are highest
# - cluster 1 appers to occupy the middle level univerisities
# - cluster 0 comes as the lowest level univerisities

# In[83]:


univ[univ['cluster_new']==0]


# In[87]:


# finding optimal k value using elbow plot
wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20),wcss)
plt.title("elbow method")
plt.xlabel('no.of clusters')
plt.ylabel('wcss')
plt.show()


# # clustering methods 
# - kmeans 
#  - kmedoids
# - kprototypes
# -hierarichal ==> no need to specify no of clusters
# - DBSCAN  ==> ""

# In[ ]:





# In[ ]:




