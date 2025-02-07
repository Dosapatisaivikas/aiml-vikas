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

# In[8]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# In[9]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='WT',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# In[10]:


fig,(ax_box,ax_hist) = plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins =30,kde=True,stat="density")
ax_hist.set(ylabel = 'Density')
plt.tight_layout()
plt.show()


# In[11]:


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

# In[13]:


cars[cars.duplicated()]


# In[14]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[15]:


cars.corr()


# ### Obseravrtions
# * The heighest corelatio is between volume and weight that is (0.999)
# * and next heighst corellation is between top speed and horse power that is(0.973)
# * lowest corelation is observed between horsepower and weight that is (0.0765)
# * Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# * Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# * Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT
# * The high coorelation among x columns is not desirable as it might lead to multicollineaity problem

# In[17]:


model = smf.ols('MPG~WT+VOL+SP+HP',data = cars).fit()


# In[18]:


model.summary()


# In[39]:


df1=pd.DataFrame()
df1['actual_y1']=cars['MPG']
df1.head()


# In[47]:


pred_y1 = model.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[49]:


from sklearn.metrics import mean_squared_error
print("MSE :",mean_squared_error(df1["actual_y1"],df1["pred_y1"]))


# In[55]:


pred_y1=model.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[57]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("mse: ",mse)
print("rmse: ",np.sqrt(mse))


# In[59]:


cars.head()


# ### checking for multicolinerality among x-columns using vif method( varince inflaton factor)

# In[61]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


#  ### observations 
#  - the ideal range of vif values shall be between 0 to 10 . however slightly higher values can be tolerated
#  - as seen from the very high vif values for vol and wt,it is clear that they are prone to multicollinearity proble
#  - hence it is decided to drop one of the ocoloumns(eithr volmor wt) to overcome the multicollinerity
#  - it is decided to drop wt and retain vol column in futher models

# In[65]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[69]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[71]:


model2.summary()


# In[87]:


df2=pd.DataFrame()
df2['actual_y2']=cars['MPG']
df2.head()


# In[89]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[93]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("mse: ",mse)
print("rmse: ",np.sqrt(mse))


# ### Observations
# - the adjusted r square value improved slightly to 0.76
# - all the p-vlaues for model parameters are less than 5% hence they are significant
# - therfore the vol,hp,sp columns are finalized as the significant predictor for the mpg response varible
# - there is no improvement in mse value

# In[96]:


cars1


# In[ ]:




