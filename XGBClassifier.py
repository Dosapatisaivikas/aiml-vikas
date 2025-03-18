#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score 
from sklearn.preprocessing import StandardScaler 
from xgboost import XGBClassifier


# In[3]:


df = pd.read_csv('diabetes.csv')
df


# In[4]:


x = df.drop('class',axis=1)
y=df['class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[5]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled)
print('-------------------------------------------------')
print(x_test_scaled)


# In[6]:


xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

param_grid = {
    'n_estimators':[100,150,200,300],
    'learning_rate': [0.01,0.1,0.15],
    'max_depth': [2,3,4,5],
    'subsample': [0.8,1.0],
    'colsample_bytree': [0.8, 1.0]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator = xgb,
                           param_grid=param_grid,
                           scoring='recall',
                           cv = skf,
                           verbose=1,
                           n_jobs=-1)


# In[13]:


grid_search.fit(x_train_scaled , y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validated Recall:", grid_search.best_score_)

y_pred = best_model.predict(x_test_scaled)


# In[15]:


print('\nconfusion matrix:\n',confusion_matrix(y_test,y_pred))
print(classification_report(y_test,Y_pred)


# In[ ]:




