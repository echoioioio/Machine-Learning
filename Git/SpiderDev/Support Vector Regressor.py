#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:32:53 2022

@author: apple
"""



# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')

# Modelling and Forecasting
# ==============================================================================
from sklearn.linear_model import Ridge
#from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


#Importing the dataset
df = pd.read_csv('dataset.csv')

#Checking the dataset first 10 rows
#df.head(10)

#Checking the dataset columns
#print(df.info())

#Rename Columns 
df.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)
df.rename(columns = {'demand [MW]':'Y'}, inplace = True)

#Check duplicated
df[df.duplicated(keep=False)]

#Drop NA
df = df.dropna(axis=0)

#Checking the dataset size after dropping NA
df.info()

#Format datetime
df.Date=pd.to_datetime(df.Date,utc=True)

# Sorting data in ascending order by the date
df = df.sort_values(by='Date')

#Set Date as index
df.set_index('Date', inplace=True)

#Check data status
print(df.info())


#Add new columns
df['Month'] = df.index.month
df['Weekday'] = df.index.weekday + 1 #in pandas, the day of the week with Monday=0, Sunday=6.
df['Hour'] = df.index.hour

# =============================================================================
# ft_train, ft_test, lb_train, lb_test = train_test_split(data , target, test_size=0.20, random_state = 2)
# display(ft_train.shape,ft_test.shape)
# =============================================================================

ft_train, ft_test= np.split(df, [int(.8 *len(df))])
lb_train, lb_test= np.split(df.Y, [int(.8 *len(df))])
print(ft_train.shape,ft_test.shape)

import numpy as np
np.random.seed(seed=5)

import warnings
warnings.simplefilter("ignore", UserWarning)

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline


# Train the model using the training sets for c=10 and e=0.4
regr = SVR(C=10, epsilon=0.4)
regr.fit(ft_train, lb_train)

# Make predictions using the testing set
lb_pred = regr.predict(ft_test)

print(lb_pred)

#predicted output values for test inputs
pred = lb_pred
# output values from the test set 
test = lb_test

#Transefer test to 2d dataframe and add index
pred = pd. DataFrame(pred, columns=['pred'])

print(pred.head())

pred = pred.set_index(test.index)

# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 3.5))
test.plot(ax=ax, linewidth=2, label='real')
pred.plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real demand')
ax.legend();


print('Support Vector Regressor')
print('------------------------------')

print('Accuracy : {}'.format(regr.score(ft_test, lb_test)))

MAE = mean_absolute_error(test, pred)
print('MAE : {}'.format(round(MAE, 2)))

MSE = mean_squared_error(test, pred)
print('MSE : {}'.format(round(MSE, 2)))

RMSE = sqrt(MSE)
print('RMSE  : %f' % RMSE)

R2_SCORE=r2_score(test, pred)
print('R2_SCORE  : %f' % R2_SCORE)

