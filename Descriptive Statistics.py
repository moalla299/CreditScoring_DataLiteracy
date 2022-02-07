#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:10:36 2022

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
from tueplots.constants import markers
from tueplots.constants.color import palettes
import csv
import os
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Font & Figuresize Check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
fonts.neurips2021()
fontsizes.neurips2021()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Directory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
os.getcwd()
MyDir = "/Users/nikki/Desktop/QDS_Tuebingen/3-Winter2021/Data Literacy/Project/CreditScoring_DataLiteracy"
os.chdir(MyDir)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Import Data
HmeqDF = pd.read_csv('/Users/nikki/Desktop/QDS_Tuebingen/3-Winter2021/Data Literacy/Project/DataSet/hmeq.csv')
#DataSet Information
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Insight into the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
HmeqDF.info()
"""
apart for the "BAD" and "LOAN" features, all other features contain missing values.
Moreover, REASON & JOB are object(String) that should be fixed.
"""
#Brief descriptive Statistics 
HmeqDF.describe()
"""
We got some statistical overview of the data
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Missing Values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
MissingData = HmeqDF.isnull().sum().rename_axis('Variables').reset_index(name='Missing Values') 
MissPerc = HmeqDF.isnull().sum()/(len(HmeqDF))*100
MissPerc = MissPerc.sort_values(ascending=False)
"""
Depending on Missing Percentage of each column's data, we saw that DEBTINC has the most missing values, 
On the other hand we guess depending on the context of this variable; it will play a major role in our classifying model. 
Later we will investigate the Feature importance and see the results

**Filling the missing values is out of this study scope, we just dropped them**
"""
#Dropna Dataset
HmeqDF_dropna = HmeqDF.dropna()
MissingData_dropna = HmeqDF_dropna.isnull().sum().rename_axis('Variables').reset_index(name='Missing Values') 
"""
Here we see that only 56% of the data will remains if we drop rowa with missing values
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Variables' Distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Columns Count Values
columnNames = pd.Series(HmeqDF.columns.values)
#Dependent Variable 
HmeqDF['BAD'].value_counts(normalize=True)
#Categorical Independent Variable (Qualitative)
HmeqDF['REASON'].value_counts(normalize=True)
HmeqDF['JOB'].value_counts(normalize=True)   #Other Maybe not that much informative
#Interval Independent Variable (Quantitative)
HmeqDF['LOAN'].value_counts()
HmeqDF['MORTDUE'].value_counts()
HmeqDF['VALUE'].value_counts()  
HmeqDF['YOJ'].value_counts()  
HmeqDF['DEROG'].value_counts()  
HmeqDF['DELINQ'].value_counts()  
HmeqDF['CLAGE'].value_counts()
HmeqDF['NINQ'].value_counts()  
HmeqDF['CLNO'].value_counts()  
HmeqDF['DEBTINC'].value_counts()
"""
We got some insights about each variable and if they are imbalanced or have outlaiers
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% All Features Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#BAD & REASON  
sns.factorplot(x='BAD', col = 'REASON',kind='violin', data=HmeqDF) #NOT INFORMATIVE 
sns.factorplot(x='BAD', col = 'REASON',kind='count', data=HmeqDF)
#BAD & JOB 
sns.factorplot(x='BAD', col = 'JOB',kind='count', data=HmeqDF)
ax = sns.swarmplot( x= HmeqDF["JOB"], y = HmeqDF["BAD"][:2000],  palette="CMRmap", size=3) # Not INFORMATIVE
#BAD & LOAN 
ax = sns.swarmplot( x= HmeqDF["JOB"], y = HmeqDF["BAD"][:2000],  palette="CMRmap", size=3) # Not INFORMATIVE
#BAD & MORTDUE 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["MORTDUE"], palette="CMRmap", size=3) # Not INFORMATIVE
#BAD & VALUE 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["VALUE"], palette="CMRmap", size=3) # Not INFORMATIVE
#BAD & YOJ 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["YOJ"], palette="CMRmap", size=3) 
"""Nice one decide to include it depending on feature importance results
   **Feature Importance Results were not in alignment with this standpoint. 
"""
#BAD & DEROG 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"][:1000], y = HmeqDF_dropna["DEROG"][:1000], palette="CMRmap", size=3) # Decide later
#BAD & DELINQ 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["DELINQ"], palette="CMRmap", size=1) # Decide later
"""Informative depending on the Feature Importance """
#BAD & CLAGE 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["CLAGE"], palette="CMRmap", size=1) # Decide later
#BAD & NINQ 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["NINQ"], palette="CMRmap", size=1) # Decide later
#BAD & CLNO 
ax = sns.swarmplot( x = HmeqDF_dropna["BAD"], y = HmeqDF_dropna["CLNO"], palette="CMRmap", size=1) # Decide later
#BAD & DebtInc 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Debt-to-Inc Ratio %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Calculating the maximum amount of DebtInc ratio for settled Credits   
SettledDF = HmeqDF [HmeqDF["BAD"] == 0]
MAXSettledDebtInc = SettledDF["DEBTINC"].max()
#Limiting the Y axis to 60 for better visualization
HmeqTest = HmeqDF_dropna[HmeqDF_dropna["DEBTINC"] <= 60]
#with plt.rc_context(bundles.neurips2021()):
style = dict(size=14 , color='black')
sns.set(rc={"figure.figsize":(12, 9)}, font='Times New Roman') 
ax = sns.swarmplot(x = HmeqTest["BAD"],  y = HmeqTest["DEBTINC"],palette="CMRmap", size=4)
ax.set_xlabel("Probability of Default or Serious Delinquency (BAD)" ,fontsize=16)
ax.set_ylabel("Debt-to-Income Ratio (DEBTINC)", fontsize= 16)
ax.legend([" Settled Credit (Class 0)", " Default or Serious Delinquency (Class 1)"], fontsize= 16)
ax.axhline(MAXSettledDebtInc, color = "gray", linestyle = '--')
ax.text(0.45 , MAXSettledDebtInc + 1.5 , "45.56", **style)
plt.savefig("Figure00000.png", format="png", dpi=1200)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Categorical Variables Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Reason: Converting to dummy variables 
REASON_dummies = pd.get_dummies(HmeqDF_dropna['REASON'], prefix = 'REAS')
HmeqDF_dropna.drop('REASON', axis=1, inplace = True)
HmeqDF_dropna["REAS_DebtCon"] = REASON_dummies ["REAS_DebtCon"]
HmeqDF_dropna["REAS_HomeImp"] = REASON_dummies ["REAS_HomeImp"]
#JOB: Frequency encoding 
HmeqDF_dropna['JOB'].value_counts()
mapper = {'Sales':0, 'Self': 1, 'Mgr': 2, 'Office': 3,  'ProfExe':4, 'Other':5 } 
data['JOB'].replace(mapper, inplace = True)
HmeqDF_dropna['JOB'].replace(mapper, inplace = True)
HmeqDF_dropna['JOB'].value_counts()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Final Processed Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Information regarding the Final Preproccesed Dataset 
HmeqDF_dropna.info()
cor = HmeqDF_dropna.corr()
#Write the clean dataset into a CSV file 
HmeqDF_dropna.to_csv('HmeqFinal.csv',index=False)

