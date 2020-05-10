# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:17:06 2020

@author: vmakm
"""
# Importing Data Analysis Librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


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




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)
# Feature Scaling


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



from sklearn.neural_network import MLPClassifier
rkf = RepeatedKFold(n_splits=3, n_repeats=3, random_state =True)
mlp = MLPClassifier()
parameters={
'hidden_layer_sizes': [(10,10,10), (10,10,10,10), (10,10,10,10,10), (10,10,10,10,10,10)],
'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]
}

clf= GridSearchCV(mlp,param_grid=parameters,scoring='roc_auc',n_jobs=-1,verbose=1,cv=rkf)
clf.fit(X, y)

result_mean = clf.cv_results_["mean_test_score"]
print(clf.cv_results_["mean_test_score"])
print(clf.best_params_)
print("Best parameter = ",clf.best_params_)
print("Best parameter = ",clf.best_score_)




