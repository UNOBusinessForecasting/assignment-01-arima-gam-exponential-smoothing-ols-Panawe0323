pip install statsmodels
pip install numpy
pip install pygam
pip install prophet
pip install ipynb
pip install numpy
pip install scipy
pip install plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import prophet
import scipy as sp
import pygam
import ipynb
from statsmodels.tsa.seasonal import seasonal_decompose

train_url = 'https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv'
train_df = pd.read_csv(train_url)

# Convert the 'Timestamp' column to datetime
train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
#train_df.set_index('Timestamp', inplace=True)

# Plot 1: Time Series of the Number of Trips
plt.figure(figsize=(12, 6))
plt.plot(train_df['trips'], label='Number of Trips')
plt.title('Number of Taxi Trips Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Trips')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Average Number of Trips per Hour
#train_df['hour'] = train_df.index.hour
hourly_avg = train_df.groupby('hour')['trips'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_avg.index, y=hourly_avg.values)
plt.title('Average Number of Taxi Trips per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Trips')
plt.grid(True)
plt.show()

# Plot 3: Average Number of Trips per Day of the Week
#train_df['day'] = train_df.index.day
daily_avg = train_df.groupby('day')['trips'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=daily_avg.index, y=daily_avg.values)
plt.title('Average Number of Taxi Trips per Day of the Week')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Number of Trips')
plt.grid(True)
plt.show()

# Plot 4: Rolling Mean and Standard Deviation
plt.figure(figsize=(12, 6))
rolling_mean = train_df['trips'].rolling(window=24).mean()  # 24-hour rolling window
rolling_std = train_df['trips'].rolling(window=24).std()

plt.plot(train_df['trips'], label='Original')
plt.plot(rolling_mean, label='Rolling Mean', color='orange')
plt.plot(rolling_std, label='Rolling Std', color='green')
plt.title('Rolling Mean & Standard Deviation of Taxi Trips')
plt.xlabel('Time')
plt.ylabel('Number of Trips')
plt.legend()
plt.grid(True)
plt.show()

# Plot 5: Decompose the time series
decomposition = seasonal_decompose(train_df['trips'], model='additive', period=24*7)  # Weekly seasonality
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

from prophet import Prophet

# Prepare data for Prophet
# 'ds' for datetime and 'y' for the target variable
prophet_df = train_df[['Timestamp', 'trips']].rename(columns={'Timestamp': 'ds', 'trips': 'y'})

# Instantiate and fit the Prophet model
model = Prophet()
model.fit(prophet_df)

# Create a future dataframe for predictions (we need to predict 744 hours into the future)
future = model.make_future_dataframe(periods=744, freq='H')

# Predict future trips
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title('Prophet Forecast for Taxi Trips')
plt.xlabel('Date')
plt.ylabel('Number of Taxi Trips')
plt.show()

# Plot components (trend, seasonality, holidays)
model.plot_components(forecast)
plt.show()

# Save model and predictions
modelFit = model
pred = forecast[['ds', 'yhat']]

test_df =  pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

test_df

# Prepare test data for predictions
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
test_df = test_df.rename(columns={'Timestamp': 'ds'})

# Generate predictions for the test dataset
future = test_df[['ds']]  # Prophet expects a dataframe with 'ds' column for future timestamps
forecast = model.predict(future)

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], forecast['yhat'], label='Predicted Trips', color='blue')
if 'trips' in test_df.columns:
    plt.plot(test_df['ds'], test_df['trips'], label='Actual Trips', color='red')
plt.title('Predicted vs Actual Taxi Trips (Test Data)')
plt.xlabel('Date')
plt.ylabel('Number of Taxi Trips')
plt.legend()
plt.grid(True)
plt.show()

# Save predictions to a dataframe
test_df['Predicted_Trips'] = forecast['yhat']

# Print the first few rows of the predictions
print(test_df[['ds', 'Predicted_Trips']].head())

# Prepare test data for Prophet
test_prophet_df = test_df[['Timestamp']].rename(columns={'Timestamp': 'ds'})
# Prepare test data for predictions
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
test_df = test_df.rename(columns={'Timestamp': 'ds'})

# Generate predictions for the test dataset
future = test_df[['ds']]  # Prophet expects a dataframe with 'ds' column for future timestamps
forecast = model.predict(future)

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], forecast['yhat'], label='Predicted Trips', color='blue')
if 'trips' in test_df.columns:
    plt.plot(test_df['ds'], test_df['trips'], label='Actual Trips', color='red')
plt.title('Predicted vs Actual Taxi Trips (Test Data)')
plt.xlabel('Date')
plt.ylabel('Number of Taxi Trips')
plt.legend()
plt.grid(True)
plt.show()

# Save predictions to a dataframe
test_df['Predicted_Trips'] = forecast['yhat']

# Print the first few rows of the predictions
print(test_df[['ds', 'Predicted_Trips']].head())













