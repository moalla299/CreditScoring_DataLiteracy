#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:40:14 2022

@author: nikki
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tueplots
from tueplots import bundles, axes
from tueplots import fonts, fontsizes
#from tueplots import figsizes
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

import csv
import os
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Directory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
os.getcwd()
MyDir = ".../CreditScoring_DataLiteracy"
os.chdir(MyDir)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Import Data
HmeqDF = pd.read_csv('HmeqPreprocessed.csv')
#DataSet Information
HmeqDF.info()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Randome Forest Classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Labels and Feature set definition
y = HmeqDF["BAD"]
x = HmeqDF.drop('BAD',axis=1)
#Train-Test dataset split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
#Fit on Train data
model.fit(x_train, y_train)
#Test Set Prediction
y_pred = model.predict(x_test)
#Probabilities for each class
probs = model.predict_proba(x_test)[:, 1]


#Merged Test and Predicted labels 
y_test1 = pd.DataFrame(y_test).reset_index()
y_pred1 = pd.DataFrame(y_pred)
y_test1["BAD_pred"] = y_pred1.iloc[1:]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Model Performance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Accuracy
print('Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
"""
93.86% percent model accuracy
"""
#ROC
roc = roc_auc_score(y_test, probs)
#Confusion Matrix
ConfMat = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', ConfMat)
#Heatmap 
Conf_matrix = pd.DataFrame(data=ConfMat, columns=['Predicted (Class 0) ', 'Predicted (Class 1)'], 
                                 index=['      (Class 0)', '     (Class 1)'])
sns.heatmap(Conf_matrix, annot=True,square= True, cbar_kws={'fraction' : 0.01}, fmt='d', cmap='Purples')
plt.savefig("Heat0.png", format="png", dpi=1200)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dummy Classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Fit on Train data
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
#Test Set Prediction
y_pred_dummy = dummy_clf.predict(x_test)
#Probabilities for each class
probs_dummy = dummy_clf.predict_proba(x_test)[:, 1]


#Merged Test and Predicted labels 
y_test1_dummy = pd.DataFrame(y_test).reset_index()
y_pred1_dummy = pd.DataFrame(y_pred_dummy)
y_test1_dummy["BAD_pred"] = y_pred1_dummy.iloc[1:]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Model Performance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Accuracy
print('Dummy Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred_dummy)))
"""
90.79% percent model accuracy
"""
#ROC
roc_dummy = roc_auc_score(y_test, probs_dummy)
#Confusion Matrix
ConfMatDummy = confusion_matrix(y_test, y_pred_dummy)
print('Confusion matrix\n\n', ConfMatDummy)
#Heatmap 
Conf_matrixDummy = pd.DataFrame(data=ConfMatDummy, columns=['Predicted (Class 0) ', 'Predicted (Class 1)'], 
                                 index=['      (Class 0)', '     (Class 1)'])
sns.heatmap(Conf_matrixDummy, annot=True,square= True, cbar_kws={'fraction' : 0.03},fmt='d', cmap='Oranges')
plt.savefig("Heat00.png", format="png", dpi=1200)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Feature importance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
Imp = pd.DataFrame({'feature': list(x.columns),'importance': model.feature_importances_}).sort_values('importance', ascending = False)
Imp.head()






