#!/usr/bin/env python
# coding: utf-8

# Importing Libraries and reading the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("C:/Users/Keerthe/Documents/ASSIGNMENT/PA PROJECT DATASET.csv")

# Display the first 10 rows of the dataset
df.head(10)

# Check information about the dataset
df.info()

# Summary statistics of the 'Price' attribute
df.describe()

# Get the number of rows and columns in the dataset
df.shape

# Display the count of null values in each column
df.isnull().sum()

# Drop rows with null values
df.dropna(inplace=True)

# Check for null values after dropping
df.isnull().sum()

# Function to change a column to datetime
def change_into_datetime(col):
    df[col] = pd.to_datetime(df[col])

# List of columns to convert to datetime
date_time_columns = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']

# Convert the specified columns to datetime
for col in date_time_columns:
    change_into_datetime(col)

# Extracting day and month from the 'Date_of_Journey' column
df['journey_date'] = df['Date_of_Journey'].dt.day
df['journey_month'] = df['Date_of_Journey'].dt.month

# Drop the 'Date_of_Journey' column
df.drop('Date_of_Journey', axis=1, inplace=True)

# Function to extract hour from datetime
def extract_hour(data, col):
    data[col + '_hour'] = data[col].dt.hour

# Function to extract minute from datetime
def extract_min(data, col):
    data[col + '_min'] = data[col].dt.minute

# Function to drop a column
def drop_col(data, col):
    data.drop(col, axis=1, inplace=True)

# Extract hour and minute from 'Dep_Time' and 'Arrival_Time' columns
extract_hour(df, 'Dep_Time')
extract_min(df, 'Dep_Time')
drop_col(df, 'Dep_Time')

extract_hour(df, 'Arrival_Time')
extract_min(df, 'Arrival_Time')
drop_col(df, 'Arrival_Time')

# Correct the 'Duration' column format
duration = list(df['Duration'])
for i in range(len(duration)):
    if len(duration[i].split(' ')) == 2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i] = duration[i] + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]

# Update the 'Duration' column
df['Duration'] = duration

# Split 'Duration' into 'dur_hour' and 'dur_min' columns
df['dur_hour'] = df['Duration'].apply(lambda x: int(x.split(' ')[0][0:-1]))
df['dur_min'] = df['Duration'].apply(lambda x: int(x.split(' ')[1][0:-1]))

# Drop the 'Duration' column
drop_col(df, 'Duration')

# Convert 'dur_hour' and 'dur_min' to integer data type
df['dur_hour'] = df['dur_hour'].astype(int)
df['dur_min'] = df['dur_min'].astype(int)

# One-hot encode the 'Airline' column
Airline = pd.get_dummies(df['Airline'], drop_first=True)

# One-hot encode the 'Source' column
source = pd.get_dummies(df['Source'], drop_first=True)

# One-hot encode the 'Destination' column
destination = pd.get_dummies(df['Destination'], drop_first=True)

# Concatenate the one-hot encoded columns and continuous columns
data = pd.concat([Airline, source, destination, df[['Total_Stops', 'journey_date', 'journey_month', 'dur_hour', 'dur_min', 'Price']]], axis=1)

# Check the first few rows of the concatenated dataframe
data.head()

# Identify and resolve outliers in the 'Price' column
def plot(data, col):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    sns.distplot(data[col], ax=ax1)
    sns.boxplot(data[col], ax=ax2)

# Plot and resolve outliers in 'Price'
plot(data, 'Price')

# Replace outliers in 'Price' with the median value
data['Price'] = np.where(data['Price'] >= 40000, data['Price'].median(), data['Price'])

# Replot the 'Price' column after removing outliers
plot(data, 'Price')

# Encode 'Total_Stops' column
dict = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}
df['Total_Stops'] = df['Total_Stops'].map(dict)

# Split data into features (X) and target (y)
X = data.drop('Price', axis=1)
y = data['Price']

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# Import necessary regression models and evaluation metrics
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define a function to train and evaluate regression models
def predict(ml_model):
    model = ml_model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2score = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return {
        'Model': str(ml_model),
        'Training score': model.score(X_train, y_train),
        'R2 score': r2score,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

# Train and evaluate different regression models
results = []

models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR()
]

for model in models:
    results.append(predict(model))

# Display the results
for result in results:
    print(result)
