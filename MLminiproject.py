#IMPORTING MODULES
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt

#READING A CSV FILE
hdp=pd.read_csv("heartdisease.csv")

#SEPERATING DEPENDENT FEATURES(Y) AND INDEPENDENT FEATURES(X)
Y=hdp.Result
X=hdp.drop(["Result"],axis=1)

#DIVIDING DATASET FOR TT(TESTING AND TRAINING)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=7140)

#CREATING A LOGISTIC REGRESSION MODEL
LoR=LogisticRegression()
LoR.fit(X,Y)

#xnew=[[57,1,2,124,261,0,0,141,0,0.3,1,0,7]]
#TESTING AND ACCURACY
y_pred=LoR.predict(x_test)
print(y_pred)
acc=accuracy_score(y_test,y_pred)
print(acc)            
        
#CONFUSION MATRIX
mat=confusion_matrix(y_pred,y_test)
print(mat)

#DUMPING MODEL/SAVING MODEL
if(joblib.dump(LoR,"TrainedModel(LoR).pkl")):
    print("Model Saved!")

