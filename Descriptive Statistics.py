#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:10:36 2022

@author: nikki
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Libraries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

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
import csv
import os

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Directory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
os.getcwd()
os.chdir('/Users/nikki/Desktop/QDS_Tuebingen/3-Winter2021/Data Literacy/Project/CreditScoring_DataLiteracy')
#import mpu.ml

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#Import Data
HmeqDF = pd.read_csv('/Users/nikki/Desktop/QDS_Tuebingen/3-Winter2021/Data Literacy/Project/DataSet/hmeq.csv')

#DataSet Information
HmeqDF.info()

"""
Except BAD & LOAN columns other columns have missing values. 
Moreover, REASON & JOB are object(String) that should be fixed. (?????????????????????????)
"""

#Missing Values 
MissingData = HmeqDF.isnull().sum().rename_axis('Variables').reset_index(name='Missing Values') 
MissPerc = HmeqDF.isnull().sum()/(len(HmeqDF))*100
MissPerc = MissPerc.sort_values(ascending=False)

"""
PAPER::::: Depending on Missing Percentage of each column's data, we can coclude that DEBTINC has the most missing values, 
On the other hand we guess depending on the meaning of this variable; it will play a major role in our classifying model. 
Later we will investigate the Feature importance and see whether it is important or not (???????????????????)
So depending on importance of some features and the huge amount of missing values, droping rows with missing values might
be not a good idea. 

NOW because it is out of this study scope we ignore the missing values. 
"""
#Dropna Dataset
HmeqDF_dropna = HmeqDF.dropna()
MissingData_dropna = HmeqDF_dropna.isnull().sum().rename_axis('Variables').reset_index(name='Missing Values') 

"""
Here we see that only 56% of the data will remains if we drop rowa with missing values
"""
#Brief descriptive Statistics 
HmeqDF.describe()

"""
We got some statistical overview of the data
"""

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Each Variable Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Check required fonts and fontsizes for the plots
fonts.neurips2021()
fontsizes.neurips2021()

#Dataset Variables
columnNames = pd.Series(HmeqDF.columns.values)

#BAD 
#HmeqDF["BAD"].apply(pd.to_numeric)
#type (HmeqDF["BAD"][1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
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
   you can ad a distribution to the plot 
   The more the Years at present job, The less the probability of defult
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

#Calculating the maximum amount of DebtInc ratio for settled Credits   
SettledDF = HmeqDF [HmeqDF["BAD"] == 0]
MAXSettledDebtInc = SettledDF["DEBTINC"].max()

#with plt.rc_context(bundles.neurips2021()):
style = dict(size=7, color='black')
ax = sns.swarmplot(x = HmeqDF_dropna["BAD"],  y = HmeqDF_dropna["DEBTINC"], palette="CMRmap",size=3)
ax.set_xlabel("Customer default or serious delinquency" ,fontsize=9)
ax.set_ylabel("Debt to income ratio", fontsize= 9)
ax.legend([" Settled Credit", " Default or serious Delinquency"], fontsize= 7)
ax.axhline(MAXSettledDebtInc, color = "gray", linestyle = '--')
ax.text(0.45 , MAXSettledDebtInc + 1.5 , "45.56", **style)
#sns.set_palette("colorblind")
plt.figure(figsize=(6,6))   
plt.rcParams.update(fonts.neurips2021())
plt.show()


"""
These plots are ugly(!!!!!!!!!!!!!!)
and the fairness is droped from this work 

Distributions of various variables
Categorical data: 
BAD = HmeqDF["BAD"].value_counts()
HmeqDF["BAD"].value_counts().plot(kind= "barh")

Reason
Reason = HmeqDF["REASON"].value_counts()
HmeqDF["REASON"].value_counts().plot(kind= "barh")

What ever is the reason of geting a credit is our model fair for both type of reasons?
Depending on their inbalanced distribution, will my model allocate fair budget to each group, since we cannot judge
who face the most essential need for the loan.
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Categorical Variables Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Reason
#Converting REASON to dummy variables 
REASON_dummies = pd.get_dummies(HmeqDF_dropna['REASON'], prefix = 'REAS')
HmeqDF_dropna.drop('REASON', axis=1, inplace = True)
HmeqDF_dropna["REAS_DebtCon"] = REASON_dummies ["REAS_DebtCon"]
HmeqDF_dropna["REAS_HomeImp"] = REASON_dummies ["REAS_HomeImp"]


#JOB
#JOB variable incoding 
HmeqDF_dropna['JOB'].value_counts()
#Incoding depend on the Frequency
mapper = {'Sales':0, 'Self': 1, 'Mgr': 2, 'Office': 3,  'ProfExe':4, 'Other':5 } 
data['JOB'].replace(mapper, inplace = True)
HmeqDF_dropna['JOB'].replace(mapper, inplace = True)
HmeqDF_dropna['JOB'].value_counts()

#Information regarding the Final Preproccesed Dataset 
HmeqDF_dropna.info()
cor = HmeqDF_dropna.corr()

#Write the clean dataset into a CSV file 
HmeqDF_dropna.to_csv('HmeqFinal.csv',index=False)








#plt.pyplot.xlim(-50,50)
DebtInc.plot.kde()

#Scatter Plot
%matplotlib inline
x = DebtInc
y = range(len(DebtInc))
plt.scatter(x, y, c=y, cmap='Spectral')
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.colorbar()
plt.title('Debt to Income Ratio')
plt.xlabel('DebtInc')
plt.ylabel('Frequency')
plt.show()


#sns.jointplot(x,y, kind="hist") "It is not bad if xlim will be more limited"
#sns.jointplot(x,y, kind="hex") #It is not bad if xlim will be more limited
fig, ax = plt.subplots()
sns.jointplot(x,y, kind="hist", color="c") 
ax.set_xlim(0,100)
ax.set_xticks(range(0,100))
plt.show()

g = sns.jointplot()
sns.scatterplot(x= x, y= y, ec="c", fc="none", linewidth= 0.5, ax=g.ax_joint)
sns.histplot(x=x, fill= False, linewidth=2 , ax=g.ax_marg_x, color=("c"))



#All the attempt for encoding the JOB, 
#because its content were sting it was problematic that I could not use MPU package. Yet I couldn't solve it 
"""import numpy as np
nb_classes = 6
x = np.array(HmeqDF["JOB"]).reshape(-1)
one_hot_targets = np.eye(nb_classes)[x]

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

X = [['Other', 1], ['ProfExe', 2], ['Office', 3], ['Mgr', 4], ['Self', 5], ['Sales', 6]]
enc.fit(X)
OrdinalEncoder()
enc.transform([['ProfExe', 3], ['Office', 1]])



enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
OrdinalEncoder()
enc.categories_

enc.transform([['Female', 3], ['Male', 1]])
array([[0., 2.],
       [1., 0.]])

HmeqDF["JOB"] = mpu.ml.indices2one_hot(len(HmeqDF["JOB"][:]), nb_classes=6)
len(HmeqDF["JOB"][1200])"""


"""Nominal data like REASON and JOB should be coded in a way that we can use them in pur model:
    1hot Encoding or categorical indexing 0-5..."""
"""The other discussable point here is: Is really category "Other" provide any helpful data for our model???"""


#Feature Importance 

#Correlations
Correl = HmeqDF.corr()['BAD'].sort_values()
