#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 03:12:22 2020

@author: vefak
"""
#Import Libraries
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score,RepeatedKFold,GridSearchCV
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#Import Libraries
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score,RepeatedKFold,GridSearchCV
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('bank-additional-full.csv', sep=";")
dataset.head()
dataset.info()
dataset.columns

# knowing the categorical variables
print('Jobs:\n', dataset['job'].unique())
print('Marital:\n', dataset['marital'].unique())
print('Default:\n', dataset['default'].unique())
print('Housing:\n', dataset['housing'].unique())
print('Loan:\n', dataset['loan'].unique())

#Trying to find some strange values or null values
print('Null Values: ', dataset.isnull().any())

# Label encoder order is alphabetical
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

categorilcals = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']

for word in (categorilcals):
    dataset[word]      = labelencoder.fit_transform(dataset[word]) 
    
   
#dataset['pdays'].describe()    
#dataset =dataset.drop('pdays',axis=1)    
    
X = dataset.iloc[:, :20].values
y = dataset.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


#Grid Search 
def gridSearchRFR(x_data,y_data,param_grid,scoring,cv):
    grid_model = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid=param_grid,
                cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
    results=grid_model.fit(x_data,y_data)
    return results
#Random Forest Regressor
def RandomForestReg(x_data,y_data,best_params,scoring,cv):
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], 
                                    n_estimators=best_params["n_estimators"],
                                    min_samples_split=best_params["min_samples_split"],
                                    min_samples_leaf=best_params["min_samples_leaf"],
                                    random_state=True, verbose=1)
    scores = cross_val_score(rfr, x_data, y_data, scoring=scoring,  cv=cv)
    return scores



# RandomForest Parameters
n_estimators = [10, 50, 100, 250, 500,1000]
max_depth = [50,150,250]
min_samples_split = [2, 3]
min_samples_leaf = [1, 2, 3]

params_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }
#Kfold
rkf = RepeatedKFold(n_splits=3, n_repeats=3, random_state =True)
print(params_grid)

# I already defined functions for Random Forest
#Call two function. First one finds best params
#Second one: Calculates score
params_out = gridSearchRFR(X,y,params_grid,scoring="roc_auc",cv=rkf)
best_param = params_out.best_params_
scores = RandomForestReg(X,y,best_param,scoring="roc_auc",cv=rkf)
