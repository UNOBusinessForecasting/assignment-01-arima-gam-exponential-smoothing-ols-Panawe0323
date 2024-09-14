# %pip install pygam
from pygam import LinearGAM, s, f
import pandas as pd
import patsy as pt
import numpy as np
from plotly import subplots
import plotly.offline as py
import plotly.graph_objs as go

# Let's load the train and test datasets
train_df = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
test_df = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")

# Preparing our training data 
X_train = train_df[['year', 'month', 'day', 'hour']]
y_train = train_df['trips']

# Preparing the test data
X_test = test_df[['year', 'month', 'day', 'hour']]

# Initializing and fitting the GAM model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))

modelFit = model.fit(X_train, y_train)

# Making predictions for the test period
pred = modelFit.predict(X_test)

# Storing the predictions in the test DataFrame and save the result
test_df['predicted_trips'] = pred

# Saving predictions to a CSV file
test_df.to_csv('predicted_taxi_trips.csv', index=False)

# printing the test DataFrame with predictions
print(test_df[['year', 'month', 'day', 'hour', 'predicted_trips']].head())
