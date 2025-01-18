#!/usr/bin/env python
# coding: utf-8

# In[15]:


def mean_value(num):
    if len(num) == 0:
        return 0
    return sum(num)/len(num)


# In[ ]:





# In[69]:


def mode_value(L):
    s = set(L)
    d={}
    for x in s:
        d[x]=L.count(x)
    m=max(d.values())
    for k in d.keys():
        if d[k]==m:
            return k
    


# In[ ]:





# 
