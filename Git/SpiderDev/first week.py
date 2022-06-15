# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:05:12 2022

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
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5


# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')



import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"

#Importing the dataset

df = pd.read_csv('dataset.csv')

df.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)

df.rename(columns = {'demand [MW]':'demand'}, inplace = True)
import pandas as pd



# Converting the Date column to the datestamp type
#df['Date'] = pd.to_datetime(df['Date'])

# Sorting data in ascending order by the date

new = df["Date"].str.split(" ", expand = True)
df["Dateonly"]= new[0]
df["Time"]= new[1]

df.Dateonly=pd.to_datetime(df.Dateonly,format='%Y-%m-%d')

df['Weekday'] = df.Dateonly.dt.weekday
df['Day'] = df.Dateonly.dt.day
df['Month'] = df.Dateonly.dt.month
df['Year'] = df.Dateonly.dt.year

new = df["Time"].str.split(":", expand = True)
df["Time"]= new[0]

df = df[(df.Year == 2017) & (df.Month == 1) & (df.Day <= 7)] 
print(df.head(40))


# Sorting data in ascending order by the date
df = df.sort_values(by='Day')

# Now, setting the Date column as the index of the dataframe
df.set_index('Day', inplace=True)

# Print the new dataframe and its summary
print(df.head(), df.describe(), sep='\n\n')
import matplotlib.pyplot as plt

# setting the graph size globally
plt.rcParams['figure.figsize'] = (30,20)


# OR You can also plot a timeseries data by the following method

df.plot(colormap='Paired', linewidth=2, fontsize=20)
plt.xlabel('First week of Jan 2017', fontsize=20)
plt.ylabel('', fontsize=20)
plt.legend(fontsize=18)
plt.fill_between()
plt.show()

         