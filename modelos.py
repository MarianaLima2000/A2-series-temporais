import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from helpers import create_dataset

def average_model(df):
    mean_value = df['Consumption'].mean()
    forecast_mean = np.full_like(df['Consumption'].iloc[-len(df):], mean_value, dtype=np.float64)
    return forecast_mean

def naive_model(df):
    forecast_naive = df['Consumption'].shift(1).dropna()
    return forecast_naive

def random_walk_model(df):
    diff = df['Consumption'].diff().dropna()
    forecast_random_walk = df['Consumption'].iloc[-1] + diff.iloc[-1]
    forecast_random_walk = np.full_like(df['Consumption'].iloc[-len(df):], forecast_random_walk, dtype=np.float64)
    return forecast_random_walk

def ses_model(df, alpha=0.2):
    model_ses = sm.tsa.SimpleExpSmoothing(df['Consumption'])
    model_ses_fit = model_ses.fit(smoothing_level=alpha, optimized=False)
    forecast_ses = model_ses_fit.forecast(len(df))
    return forecast_ses

def holt_model(df):
    model_holt = sm.tsa.Holt(df['Consumption'])
    model_holt_fit = model_holt.fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    forecast_holt = model_holt_fit.forecast(len(df))
    return forecast_holt

def holt_winters_add_model(df):
    model_hw_add = sm.tsa.ExponentialSmoothing(df['Consumption'], trend='add', seasonal='add', seasonal_periods=4)
    model_hw_add_fit = model_hw_add.fit()
    forecast_hw_add = model_hw_add_fit.forecast(len(df))
    return forecast_hw_add

def holt_winters_mul_model(df):
    model_hw_mul = sm.tsa.ExponentialSmoothing(df['Consumption'], trend='add', seasonal='mul', seasonal_periods=4)
    model_hw_mul_fit = model_hw_mul.fit()
    forecast_hw_mul = model_hw_mul_fit.forecast(len(df))
    return forecast_hw_mul

def mult_linear_regression(df):
    X = df[['Income', 'Production', 'Savings', 'Unemployment']]
    y = df['Consumption']
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = sm.OLS(y_train, X_train).fit()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return y_train, y_test, y_train_pred, y_test_pred

def sarima_model(df):
    p, d, q = 2, 0, 2 
    P, D, Q, s = 1, 1, 1, 4 
    train, test = train_test_split(df['Consumption'], test_size=0.2, shuffle=False) 
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    sarima_model = model.fit(disp=False)
    y_pred = sarima_model.forecast(steps=len(test))
    residuals = test - y_pred

    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label='Resíduos', color='red')
    plt.axhline(0, linestyle='--', color='black', linewidth=1)
    plt.title('Resíduos ao Longo do Tempo')
    plt.xlabel('Tempo')
    plt.ylabel('Erro')
    plt.legend()
    plt.show()

    plot_acf(residuals, lags=39, title='Autocorrelação dos Resíduos')
    plt.show()
    return test, y_pred


def conv_1d(df):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Consumption'] = scaler.fit_transform(df[['Consumption']])

    train, test = train_test_split(df['Consumption'], test_size=0.2, shuffle=False)
    time_step = 10  # Ajuste o tamanho da janela temporal
    X_train, y_train = create_dataset(train, time_step)
    X_test, y_test = create_dataset(test, time_step)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    return predictions_rescaled, y_test_rescaled

