# -*- coding: utf-8 -*-
"""
Name: Arvind Kumar Gupta
Roll No.: 18AT91R02
Assignment No.: 5


Created on Tue Oct 23 16:55:30 2018

"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#%% read data from file
dataSet = pd.read_csv('spambase.csv', header=None)
dataSet = dataSet.values
dataSize = dataSet.shape

#%% separate attributes and class level
attributrLevel = dataSet[:,0:57]
classLevel = dataSet[:,57]

#%% split train and testdata   
attributrLevel_train, attributrLevel_test, classLevel_train, classLevel_test = train_test_split(attributrLevel, classLevel, test_size = 0.30)  

#%% fit SVM classifier with kernel  
genC = [0.1, 0.5, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

for kernelD in ('rbf', 'linear', 'poly'):
    print(kernelD)
    for gC in genC:
        svclassifier = SVC(C=gC, kernel=kernelD, degree=2) 
        svclassifier.fit(attributrLevel_train, classLevel_train) 
        #%% predict the testdata
        y_pred = svclassifier.predict(attributrLevel_test) 
        print(accuracy_score(classLevel_test, y_pred))

    
