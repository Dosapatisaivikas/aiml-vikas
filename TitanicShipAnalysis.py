#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install mlxtend')


# In[4]:


import pandas as pd 
import mlxtend 
from mlxtend.frequent_patterns import apriori, association_rules 
import matplotlib.pyplot as plt 


# In[6]:


titanic = pd.read_csv("Titanic.csv")
titanic 



# In[8]:


titanic.info()


# ### Observations
# - There are no null values
# - All columns are object data type and categorical in nature
# - As the columns are categorical , we can adopt one-hot-encoding 

# In[13]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index , counts.values)


# In[15]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index , counts.values)


# In[17]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index , counts.values)


# In[19]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index , counts.values)


# ### Observations
# - There are many Adult and less children
# - The highest numbers of peoples present are crew members
# - Very less people have survived

# In[22]:


df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[26]:


df.info()


# In[28]:


frequent_itemsets = apriori(df, min_support = 0.05, use_colnames= True, max_len=None)
frequent_itemsets


# In[60]:


frequent_itemsets.info()


# In[62]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules 


# In[91]:


rules.sort_values(by='lift',ascending=False)


# ### Conclusion 
# - Adult Females travelling in 1st class survived most.

# In[96]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




