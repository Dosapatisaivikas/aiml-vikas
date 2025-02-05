#!/usr/bin/env python
# coding: utf-8

# #### Assumptions iin multilinear regression 
# * Linearity: The relation betweeen the predictor and response in linear
# * Independence: Obsevation are independent of each other.
# * Homoscedasyicity: The residuals (Y-Y_hat)exhibit constant varience at all levels of the predictor.
# * Normal Distribution of Errors: The residuals of the model are normally distributed.
# * No multicollinearlity: The independent variable shouls not be too highly correlated with each other

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[14]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[16]:


cars=pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ## Descrioption
# * MPG: Mileage of car(mile per gallon)(this is y columns to be predicted)
# * HP: horse power
# * vol: volume of car
# * sp: top spped of car
# * wt: weight of car

# In[21]:


cars.info()
cars.isnull().sum()


# ### Observations
# * there are no missing values
# * there are 81 observations
# * the data types of the columns are also relevant and valid

# In[ ]:




