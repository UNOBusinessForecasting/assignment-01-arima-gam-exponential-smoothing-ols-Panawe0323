# Import required libraries
from pygam import LinearGAM, s
import pandas as pd
import numpy as np
import patsy as pt
from plotly import subplots
import plotly.offline as py
import plotly.graph_objs as go

# Load the training and test datasets
train_df = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
test_df = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

# Prepare the equation for the GAM model: using hour, day, month, and year as smooth terms
# 'trips' is the target variable, and we use hour, day, and other features as predictors
eqn = "trips ~ -1 + s(hour) + s(day) + s(month) + year"

# Generate y (dependent variable) and X (independent variables) matrices
y, X = pt.dmatrices(eqn, data=train_df)

# Initialize and fit the model (using smooth terms for the features)
model = LinearGAM(s(0) + s(1) + s(2) + s(3))

modelFit = model.gridsearch(np.asarray(X), y)

# Plot partial dependence plots for the variables
titles = ['hour', 'day', 'month', 'year']
fig = subplots.make_subplots(rows=2, cols=2, subplot_titles=titles)
fig['layout'].update(height=800, width=1200, title='Partial Dependence Plots for GAM', showlegend=False)

# Plot the partial dependence for each term
for i, title in enumerate(titles):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=.95)
    
    trace = go.Scatter(x=XX[:, i], y=pdep, mode='lines', name='Effect')
    ci1 = go.Scatter(x=XX[:, i], y=confi[:, 0], line=dict(dash='dash', color='grey'), name='95% CI')
    ci2 = go.Scatter(x=XX[:, i], y=confi[:, 1], line=dict(dash='dash', color='grey'), name='95% CI')

    # Arrange the plots in the grid
    if i < 2:
        fig.append_trace(trace, 1, i + 1)
        fig.append_trace(ci1, 1, i + 1)
        fig.append_trace(ci2, 1, i + 1)
    else:
        fig.append_trace(trace, 2, i - 1)
        fig.append_trace(ci1, 2, i - 1)
        fig.append_trace(ci2, 2, i - 1)

py.plot(fig)

# Making a Forecast for the test dataset (January 2019)
# We need to align the new test data with the same feature set
test_eqn = "trips ~ -1 + s(hour) + s(day) + s(month) + year"
_, X_test = pt.dmatrices(test_eqn, data=test_df)

# Predict on the test data
pred = gam.predict(X_test)
