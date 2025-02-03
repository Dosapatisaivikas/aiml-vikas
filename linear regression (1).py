#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf


# In[12]:


data1 = pd.read_csv("NewspaperData.csv")
print(data)


# In[14]:


data1.info()


# In[16]:


data1.isnull().sum()


# In[18]:


data .describe()


# In[24]:


plt.figure(figsize=(6,3))
plt.title("box plot for daily sales")
plt.boxplot(data1["daily"],vert=False)
plt.show()


# In[26]:


sns.histplot(data1["daily"],kde = True,stat='density',)
plt.title ("daily sales")
plt.show()


# In[28]:


plt.hist(data1["daily"])
plt.show()


# In[30]:


plt.figure(figsize=(6,3))
plt.title("box plot for Sunday")
plt.boxplot(data1["sunday"],vert=False)
plt.show()


# In[32]:


sns.histplot(data1["sunday"],kde=True,stat='density')
plt.title("sunday ")
plt.show()


# #### Observations
# * There are no missing values
# * Both the daily,sunday are Right-skewed
# * There are two outliers in each 

# #### Scatter plot and Correlation Strength

# In[36]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0,max(x) + 100)
plt.ylim(0,max(y) + 100)
plt.show()


# In[38]:


data1["daily"].corr(data1["sunday"])


# In[40]:


data1[["daily","sunday"]].corr()


# # observations
# * The relation between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# * The Correlation is strong positive with pearsons  correlation pf 0.958154

# In[43]:


# build regresion model
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[45]:


model1.summary()


# In[57]:


x = data1["daily"].values
y =data1["sunday"].values
plt.scatter(x,y,color = "m",marker = 'o',s=30)
b0 =13.84
b1 = 1.33
y_hat = b0 + b1*x
plt.plot(x,y_hat,color = "g")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[59]:


sns.regplot(x='daily', y='sunday', data=data1)
plt.xlim([0,1250])
plt.show()


# In[63]:


newdata=pd.Series([200,300,1500])


# In[67]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[69]:


model1.predict(data_pred)


# 
# pred = model1.predict(data1["daily"])
# pred

# In[75]:


data1["y_hat"] = pred


# In[79]:


data1["residuals"] = data1["sunday"]-data1["y_hat"]
data1


# In[85]:


mse = np.mean((data1["daily"]-data1["y_hat"])**2)
rmse = np.sqrt(mse)
print("rmse: ",rmse)
print("mse: ",mse)


# In[87]:


mae = np.mean(np.abs(data1["daily"]-data1["y_hat"]))
mae


# In[93]:


plt.scatter(data1["y_hat"],data1["residuals"])


# In[ ]:




