#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:40:14 2022

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#Import Data
HmeqDF = pd.read_csv('/Users/nikki/Desktop/QDS_Tuebingen/3-Winter2021/Data Literacy/Project/CreditScoring_DataLiteracy/HmeqFinal.csv')

#DataSet Information
HmeqDF.info()