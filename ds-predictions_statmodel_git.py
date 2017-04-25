

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm


n = 40000
def run():
    fp = 'data/turnstile_data_master_with_weather.csv'
    df = pd.read_csv(fp, nrows=n)
    prediction = predictions(df)
    print(prediction)


def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    if (sigma == 0).any():
        raise Exception("Feature with same value in all dataset")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma


def predictions(df_in):
    # select the features to use
    #feature_names = ['meantempi', 'Hour']
    feature_names = ['Hour', 'maxpressurei','maxdewpti','mindewpti', 'minpressurei','meandewpti','meanpressurei', 'meanwindspdi','mintempi','meantempi', 'maxtempi','precipi']

    # initialize the Y values
    X = sm.add_constant(df_in[feature_names])
    Y = df_in['ENTRIESn_hourly']

    # initialize the X features by add dummy units, standardize, and add constant
    dummy_units = pd.get_dummies(df_in['UNIT'], prefix='unit')
    X = df_in[feature_names].join(dummy_units)
    X, mu, sigma = normalize_features(X)

    # add constant in model will improve a little bit
    X = sm.add_constant(X)

    # ordinary least squares model
    model = sm.OLS(Y, X)

    # fit the model
    results = model.fit()
    # print results.summary()
    print('R^2 = ', results.rsquared)



    prediction = results.predict(X)

    return prediction


run()