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
    
   
dataset['pdays'].describe()    
dataset =dataset.drop('pdays',axis=1)    
    
X = dataset.iloc[:, :19].values
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
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state =True)
params=np.array([0.0001,0.0005,0.001,0.002,0.005,0.01,0.05,0.1,0.5,0.75,1.0,5.0,10.0,50.0,100.0,500.0,1000.0,5000.0,7500.0,10000.0])
lr = LogisticRegression()
parameters = {'C': params}
clf = GridSearchCV(lr, parameters, scoring='roc_auc', cv = rkf,n_jobs=-1)
scores=clf.fit(X, y)








prob = clf.predict(X_test)
lr_probs = clf.predict(X_test)

lr_auc = roc_auc_score(y_test, prob)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
print('Logistic: ROC AUC=%.3f' % (lr_auc))
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')




import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

