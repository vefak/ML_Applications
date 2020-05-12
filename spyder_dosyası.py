

#Import Libraries
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score,RepeatedKFold,GridSearchCV
import numpy as np
import matplotlib as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def findOptimalRidgeAlpha(x_data,y_data,parameters,scoring,cv):
    ridge_model = Ridge(normalize=True)
    ridgeRegressor = GridSearchCV(ridge_model,parameters,scoring=scoring,cv=cv)
    ridgeRegressor.fit(x_data,y_data)
    
    return ridgeRegressor.best_params_, ridgeRegressor.best_score_, ridgeRegressor

#Data import
data = pd.read_excel("ENB2012_data.xlsx")
X = data.iloc[:, 0:8].values
y = data.iloc[:, 8:10].values
y1= y[:,0]
y2= y[:,1]


#Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
y1_train, y1_test = y_train[:,0], y_test[:,0]
y2_train, y2_test = y_train[:,1], y_test[:,1]




parameters = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}



print("Y1: ")
#Optimal parameter for mean_sqauared_error
best_alpha,best_scr,ridgeRegressor = findOptimalRidgeAlpha(X_train,y1_train,parameters,"neg_mean_squared_error",10)
print("Mean Squared Error: ")
print("Optimal Alpha= {}, Best Score= {}\n".format(best_alpha,best_scr))

#Optimal parameter for mean_abosule_error
best_alpha,best_scr,ridgeRegressor = findOptimalRidgeAlpha(X_train,y1_train,parameters,"neg_mean_absolute_error",10)
print("Mean Absolute Error:")
print("Optimal Alpha= {}, Best Score= {}".format(best_alpha,best_scr))
print("----------------------------------------------------------------------")


print("Y2: ")
#Optimal parameter for mean_sqauared_error
best_alpha,best_scr,ridgeRegressor = findOptimalRidgeAlpha(X_train,y2_train,parameters,"neg_mean_squared_error",10)
print("Mean Squared Error: ")
print("Optimal Alpha= {}, Best Score= {}\n".format(best_alpha,best_scr))

#Optimal parameter for mean_abosule_error
best_alpha,best_scr,ridgeRegressor = findOptimalRidgeAlpha(X_train,y2_train,parameters,"neg_mean_absolute_error",10)

print("Mean Absolute Error:")
print("Optimal Alpha= {}, Best Score= {}".format(best_alpha,best_scr))
print("----------------------------------------------------------------------")

ridge = Ridge(alpha=0.001,normalize=True)
ridge.fit(X_train, y_train)
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state =True)
scores_1 = cross_val_score(ridge,X_test,y1_test,scoring="neg_mean_squared_error",cv=rkf)
scores_1.mean()
#scores.std(axis=0)
scores_2 = cross_val_score(ridge,X_test,y1_test,scoring="neg_mean_absolute_error",cv=rkf)



model = Ridge(alpha=0.001,normalize=True)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X_test, y1_test, cv=rkf, scoring=scoring)
print(results.mean())

