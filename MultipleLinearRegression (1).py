#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in multilinear regression 
# * Linearity: The relation betweeen the predictor and response in linear
# * Independence: Obsevation are independent of each other.
# * Homoscedasyicity: The residuals (Y-Y_hat)exhibit constant varience at all levels of the predictor.
# * Normal Distribution of Errors: The residuals of the model are normally distributed.
# * No multicollinearlity: The independent variable shouls not be too highly correlated with each other

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars=pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ## Descrioption
# * MPG: Mileage of car(mile per gallon)(this is y columns to be predicted)
# * HP: horse power
# * vol: volume of car
# * sp: top spped of car
# * wt: weight of car

# In[6]:


cars.info()
cars.isnull().sum()


# ### Observations
# * there are no missing values
# * there are 81 observations
# * the data types of the columns are also relevant and valid

# In[18]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# In[20]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='WT',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# In[22]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# In[26]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='SP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# 
# ## Observations
# - Hp is right skewed and having 7 outliers
# - Sp is right skewed and having 6 outliers
# - VOL and WT are slightly skewed and having less outliers
# - THe extrme values of cars data may have come from the specially designes natue of cars
# - as this is multidimensional data ,outliers with respect to spatial dimensions may have to be  considered while building the regression model

# In[29]:


cars[cars.duplicated()]


# In[31]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[33]:


cars.corr()


# ### Obseravrtions
# * The heighest corelatio is between volume and weight that is (0.999)
# * and next heighst corellation is between top speed and horse power that is(0.973)
# * lowest corelation is observed between horsepower and weight that is (0.0765)
# * Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# * Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# * Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT
# * The high coorelation among x columns is not desirable as it might lead to multicollineaity problem

# In[48]:


model = smf.ols('MPG~WT+VOL+SP+HP',data = cars).fit()


# In[50]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




