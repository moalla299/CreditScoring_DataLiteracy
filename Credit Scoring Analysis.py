#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 06:28:53 2022

@author: nikki
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Credit Scoring Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
"""
The aim of this study was not to train a best classifier for Hmeq DataSet, but it's aim is to shed light on 
post classification analysis that play a critical role in sensitive applications like Banking and Finance Sector
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Randome Forest Classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#To get rows with False Negative Prediction in the Test data (Random Forest Classifier)
FalseNegative = y_test1.loc[(y_test1['BAD'] == 1) & (y_test1['BAD_pred'] == 0)]
FNList = FalseNegative["index"]
FalseNegativeDf = HmeqDF.iloc[FNList, :]

#Asset Under Risk
TotalCredit = np.sum(HmeqDF["LOAN"])
RiskyCredit = np.sum(FalseNegativeDf["LOAN"])
AssetUnderRisk = RiskyCredit/TotalCredit

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dummy Classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#To get rows with False Negative Prediction in the Test data (Dummy Classifier)
FalseNegativeDummy = y_test1dummy.loc[(y_test1dummy['BAD'] == 1) & (y_test1dummy['BAD_pred'] == 0)]
FNListDummy = FalseNegativeDummy["index"]
FalseNegativeDummyDf = HmeqDF.iloc[FNListDummy, :]

#Asset Under Risk
TotalCredit = np.sum(HmeqDF["LOAN"])
RiskyCreditDummy = np.sum(FalseNegativeDummyDf["LOAN"])
AssetUnderRiskDummy = RiskyCreditDummy/TotalCredit

#Some Comparisons
np.sum (y_test == 1)
np.sum (y_pred == 1)


y_test_sub = [y_test1[y_test1["BAD"]==1]]
y_test_sub = np.array(y_test_sub)
y_test_sub1 = pd.DataFrame(y_test_sub.reshape(88 , 3))
(y_test_sub1 == 0).sum(axis=0)
