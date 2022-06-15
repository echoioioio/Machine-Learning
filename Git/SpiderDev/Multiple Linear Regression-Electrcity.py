# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:53:03 2022

@author: B1475063
"""


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

#from brokenaxes import brokenaxes
from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import warnings 
warnings.filterwarnings('ignore')

#Importing the dataset

df = pd.read_csv('dataset.csv')

df.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)

df.rename(columns = {'demand [MW]':'demand'}, inplace = True)

new = df["Date"].str.split(" ", expand = True)
df["Date"]= new[0]
df["Time"]= new[1]

df.Date=pd.to_datetime(df.Date,format='%Y-%m-%d')

df['Weekday'] = df.Date.dt.weekday
df['Month'] = df.Date.dt.month
df['Year'] = df.Date.dt.year
df.drop(columns =["Date"], inplace = True)


new = df["Time"].str.split("+", expand = True)
df["Time"]= new[0]
df["TimeZone"]= new[1]

new = df["Time"].str.split(":", expand = True)
df["Time"]= new[0]

new = df["TimeZone"].str.split(":", expand = True)
df["TimeZone"]= new[0]
 
#df.Time=pd.to_datetime(df.Time, format='%H:%M:%S')

#df.drop(columns =["Time"], inplace = True)
#df.drop(columns =["TimeZone"], inplace = True)

df = df.dropna(axis=0)
# =============================================================================
# 
#print(df.head())
#print(df.info())
# rDF = df.corr()
# print(rDF)
# #sns.pairplot(df,x_vars=["air_tmp [Kelvin]","Month"],y_vars="demand",size=7,aspect=0.8,kind="reg")
# sns.pairplot(df,x_vars=["radiation [W/m2]","Weekday"],y_vars="demand",size=7,aspect=0.8,kind="reg")
# plt.show
# target = df.values[1::2, 2]
# print(target)
# data = np.hstack([df.values[::2, :], df.values[1::2, :2]])
# =============================================================================

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model, metrics

# load the df dataset


# defining feature matrix(X) and response vector(y)
#X = df[list(df.columns)]
#print(X)
X = df[["Time","apparent_tmp [Kelvin]","Month",'radiation [W/m2]',
        "Weekday","albedo [%]","air_tmp [Kelvin]","solar_actual [MW]",
        "wind_actual [MW]","ground_tmp [Kelvin]",
        "solar_forecast [MW]","TimeZone"]]
#print(X)
y = df['demand']


# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
													random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))
print("MAE",mean_absolute_error(y_test,reg.predict(X_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
			color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
			color = "blue", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()


