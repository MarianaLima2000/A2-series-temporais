import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox



def MSE(y_true, y_pred):
    mae = mean_squared_error(y_true, y_pred)
    return mae

def RMSE(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def normalize_data(df):
    scaler = MinMaxScaler()
    df[['Consumption', 'Income', 'Production', 'Savings', 'Unemployment']] = scaler.fit_transform(df[['Consumption', 'Income', 'Production', 'Savings', 'Unemployment']])
    return df

def log_transform_data(df):
    df['Consumption'] = np.log(df['Consumption'] + 1)
    df['Income'] = np.log(df['Income'] + 1)
    df['Production'] = np.log(df['Production'] + 1)
    df['Savings'] = np.log(df['Savings'] + 1)
    df['Unemployment'] = np.log(df['Unemployment'] + 1)
    return df

def differentiation_transform_data(df):
    df = df.diff().dropna()
    return df

def dickey_fuller_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    return result

def breusch_pagan_test(df):
    X = sm.add_constant(df[['Income', 'Production', 'Savings', 'Unemployment']])
    y = df['Consumption']
    model = sm.OLS(y, X).fit()
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan test p-value: {bp_test[1]}")

def evaluate_decomposition(df):
    result = seasonal_decompose(df['Consumption'], model='additive', period=4)
    result.plot()
    plt.show()
    result.resid.dropna(inplace=True)
    result_resid_adf = adfuller(result.resid)
    print(f"ADF Statistic (Resíduos): {result_resid_adf[0]}")
    print(f"p-value (Resíduos): {result_resid_adf[1]}")

def ACF_graph(df):
    plot_acf(df['Consumption'], lags=40)
    plt.title("Função de Autocorrelação (ACF) para Consumo")
    plt.show()

def PACF_graph(df):
    plot_pacf(df['Consumption'], lags=40)
    plt.title("Função de Autocorrelação Parcial (PACF) para Consumo")
    plt.show()

def ljung_box_test(df, lags=20):
    ljung_box_result = acorr_ljungbox(df['Consumption'], lags=lags)
    print(ljung_box_result)
