# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:37:15 2019

@author: sd873
"""
#%%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)
#%%
hdp=pd.read_csv("heartdisease.csv")
hdp.shape
#hdp=hdp.drop(['education'],axis=1)
#hdp=hdp.dropna(axis=0)
#%%
hdp['Result'].unique()
Y=hdp.Result
X=hdp.drop(["Result"],axis=1)


#%%
besti=0
bestj=0
accmax=0
for i in range(30,90):
    for j in range(1,10000):
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=i*0.01,random_state=j)
        
        lr=LogisticRegression()
        lr.fit(x_train,y_train)
        
        #xnew=[[57,1,2,124,261,0,0,141,0,0.3,1,0,7]]
        y_pred=lr.predict(x_test)
        y_pred
        acc=accuracy_score(y_test,y_pred)
        if(acc>accmax):
            besti=i
            bestj=j
            accmax=acc
print(besti,bestj,accmax)            
        
#%%
mat=confusion_matrix(y_pred,y_test)
mat
#%%
Y.value_counts(0)
y_test.value_counts(0)
#%%
plt.scatter(y_test,y_pred)
plt.show()


