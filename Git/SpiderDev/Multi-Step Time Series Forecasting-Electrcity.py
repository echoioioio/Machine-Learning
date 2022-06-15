# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 14:46:51 2022

@author: b1475063
"""

# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5


# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from joblib import dump, load

import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"



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



# Split data into train-test
# ==============================================================================
steps = 36161 #The last 36 months are used as the test set to evaluate the predictive capacity of the model.
data_train = df[:-steps]
data_test  = df[-steps:]

fig, ax=plt.subplots(figsize=(9, 4))
data_train['Y'].plot(ax=ax, label='train')
data_test['Y'].plot(ax=ax, label='test')
ax.legend();



# Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 5
                )



#y = y.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
fore = forecaster.fit(y=data_train['Y'])

print("fore type", type(fore))


#print(fore.head())


# Predictions
# ==============================================================================
steps = 36161
predictions = forecaster.predict(steps=steps)
#print("type(predictions)",type(predictions))
predictions = pd.DataFrame(predictions)
#print("type(predictions)",type(predictions))
print(predictions.head())
# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_train['Y'].plot(ax=ax, label='train')
data_test['Y'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend();

# Error
# ==============================================================================
error_mse = mean_squared_error(
                y_true = data_test['Y'],
                y_pred = predictions
                )
print(error_mse)
