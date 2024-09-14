from pygam import LinearGAM, s
import pandas as pd
import numpy as np
import patsy as pt

# Loading the training and test datasets
train_df = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
test_df = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

# Prepare the equation for the GAM model: using hour, day, month, and year as smooth terms
# 'trips' is the dependent variable, and we use 'hour', 'day', 'month', and 'year' as predictors
eqn = "trips ~ -1 + s(hour) + s(day) + s(month) + year"

# Generate y (dependent variable) and X (independent variables) matrices using patsy
y, X = pt.dmatrices(eqn, data=train_df)

# Step 1: Initialize the model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))

# Step 2: Fit the model to the training data
modelFit = model.gridsearch(np.asarray(X), y)

# Step 3: Prepare the test data for prediction
# Create the same formula for the test data and generate the X matrix
_, X_test = pt.dmatrices(eqn, data=test_df)

# Step 4: Predict the number of trips in the test period (January 2019, 744 hours)
pred = modelFit.predict(X_test)

# Output the first 10 predicted values as a preview
print(pred[:10])
