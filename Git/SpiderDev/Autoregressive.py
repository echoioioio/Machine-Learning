#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:06:50 2022

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


# Autocorrelation plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3))
plot_acf(df.Y, ax=ax, lags=60)
plt.show()


# Partial autocorrelation plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3))
plot_pacf(df.Y, ax=ax, lags=60)
plt.show()




from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Identifiy Y
x = "Y"

## split data
df_train, df_test= np.split(df, [int(.8 *len(df))])



## print info
print("X_train shape:", df_train.drop("Y",axis=1).shape, "| X_test shape:", df_test.drop("Y",axis=1).shape)
print("y_train mean:", round(np.mean(df_train["Y"]),2), "| y_test mean:", round(np.mean(df_test["Y"]),2))
print(df_train.shape[1], "features:", df_train.drop("Y",axis=1).columns.to_list())



# Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = Ridge(normalize=True),
                lags      = 24 #previous 24 hours are used as predictors
             )

forecaster.fit(y=df_train.Y)
#forecaster

#print("type",df.index.inferred_type == "datetime64")

# Backtest
# ==============================================================================
metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = df.Y,
                            initial_train_size = len(df_train),
                            steps      = 24,
                            #metric     = 'mean_absolute_error',
                            metric     = 'mean_absolute_percentage_error',
                            verbose    = True
                        )

predictions = predictions.set_index(df_test.index)




# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 3.5))
df.loc[df_test.index, 'Y'].plot(ax=ax, linewidth=2, label='real')
predictions.plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real Y')
ax.legend();


# Backtest error
# ==============================================================================


# =============================================================================
# ## Kpi


print("R2 (explained variance):", round(metrics.r2_score(df_test["Y"], predictions), 3))
print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(df_test["Y"], predictions)))
print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(df_test["Y"], predictions))))
print(f'Mean Absolute Perc Error : {metric}')
 
# 
# =============================================================================
# Hyperparameter Grid search
# ==============================================================================
