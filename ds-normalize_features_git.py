import math
import numpy as np
import pandas
from ggplot import *


"""
In this tutorial, we will implement linear regression to predict ENTRIESn_hourly with R has to be > 0.4
"""

def run():
    fp = 'data/turnstile_data_master_with_weather.csv'
    df = pandas.read_csv(fp, nrows = 15000)

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    f_data, plot, r = predictions(df)
    print('r^2:', r**2)
    plot.show()


def normalize_features(df):

    mu = df.mean()
    sigma = df.std()
    if (sigma == 0).any():
        raise Exception("Feature with same value in all dataset.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_cost(features, values, theta):

    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2 * m)
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):

    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        predict_v = np.dot(features, theta)
        theta = theta - alpha / m * np.dot((predict_v - values), features)
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)


def predictions(dataframe):


    # Select Features (try different features!) - all feature to predict ENTRIESn_hourly]
    features = dataframe[['Hour', 'maxpressurei','maxdewpti','mindewpti', 'minpressurei','meandewpti','meanpressurei', 'meanwindspdi','mintempi','meantempi', 'maxtempi','precipi']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values - or y in model
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    # Add a column of 1s (y intercept)
    features['ones'] = np.ones(m)

    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    # learning rate
    alpha = 0.1
    # # of data set want to try
    num_iterations = 15000

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array,
                                                            values_array,
                                                            theta_gradient_descent,
                                                            alpha,
                                                            num_iterations)

    #plot cost history
    plot = None
    # Uncomment the next line to run faster
    plot = plot_cost_history(alpha, cost_history)

    #the predictions is outcome we predict.
    # this is y predict (not real y)
    predictions = np.dot(features_array, theta_gradient_descent)

    r = math.sqrt(compute_r_squared(values_array, predictions))
    # the set of theta after gradient descent
    #print('theta_descent:', theta_gradient_descent)
    return predictions, plot, r


def plot_cost_history(alpha, cost_history):

    cost_df = pandas.DataFrame({
        'Cost_History': cost_history,
        'Iteration': range(len(cost_history))
    })

    return ggplot(cost_df, aes(x='Iteration', y='Cost_History')) + \
            geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha)


def compute_r_squared(data, predictions):

    top = (np.square(data - predictions)).sum()
    bottom = (np.square(data - np.mean(data))).sum()


    return 1 - top/bottom

run()