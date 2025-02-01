#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf


# In[27]:


data1 = pd.read_csv("NewspaperData.csv")
print(data)


# In[29]:


data1.info()


# In[31]:


data1.isnull().sum()


# In[34]:


data .describe()


# In[36]:


plt.figure(figsize=(6,3))
plt.title("box plot for daily sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[54]:


sns.histplot(data1["daily"],kde = True,stat='density',)
plt.title ("daily sales")
plt.show()


# In[56]:


plt.hist(data1["daily"])
plt.show()


# In[60]:


plt.figure(figsize=(6,3))
plt.title("box plot for Sunday")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# In[64]:


sns.histplot(data1["sunday"],kde=True,stat='density')
plt.title("sunday ")
plt.show()


# #### Observations
# * There are no missing values
# * Both the daily,sunday are Right-skewed
# * There are two outliers in each 

# #### Scatter plot and Correlation Strength

# In[74]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0,max(x) + 100)
plt.ylim(0,max(y) + 100)
plt.show()


# In[76]:


data1["daily"].corr(data1["sunday"])


# In[78]:


data1[["daily","sunday"]].corr()


# # observations
# * The relation between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# * The Correlation is strong positive with pearsons  correlation pf 0.958154

# In[83]:


# build regresion model
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[85]:


model1.summary()


# 
