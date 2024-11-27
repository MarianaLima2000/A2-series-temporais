import pandas as pd
import numpy as np

from helpers import convert_quarter_to_date
from pre_analise import (dickey_fuller_test, breusch_pagan_test, normalize_data, 
                         ACF_graph, evaluate_decomposition, ljung_box_test, PACF_graph,
                         MSE, RMSE, MAPE, log_transform_data)
from modelos import (average_model, naive_model, random_walk_model, ses_model, holt_model,
                     holt_winters_add_model, holt_winters_mul_model, mult_linear_regression,
                     sarima_model, conv_1d)

# Carregando os dados
df = pd.read_csv("us_change.csv")
df['Quarter'] = df['Quarter'].apply(convert_quarter_to_date)
df.set_index('Quarter', inplace=True)


# Testando Necessidade de Transformação (estacionariedade)
dickey_fuller_test(df['Consumption'])

# Testando Homocedasticidade
breusch_pagan_test(df)
# Normalizando dados pelo p-value do teste
df = normalize_data(df)

# Decompondo a série e avaliando resíduos
evaluate_decomposition(df)

# Analisando sazonabilidade
ACF_graph(df)
PACF_graph(df)
ljung_box_test(df)

# Modelos Baseline
y_pred = average_model(df)
mse_mean = MSE(df['Consumption'], y_pred)
rmse_mean = RMSE(df['Consumption'], y_pred)

y_pred = naive_model(df)
mse_naive = MSE(df['Consumption'].iloc[1:], y_pred)
rmse_naive = RMSE(df['Consumption'].iloc[1:], y_pred)

y_pred = random_walk_model(df)
mse_random_walk = MSE(df['Consumption'], y_pred)
rmse_random_walk = RMSE(df['Consumption'], y_pred)

print(f"MSE (Modelo de Média): {mse_mean}")
print(f"RMSE (Modelo de Média): {rmse_mean}")
print(f"MSE (Modelo Naive): {mse_naive}")
print(f"RMSE (Modelo Naive): {rmse_naive}")
print(f"MSE (Modelo Random Walk): {mse_random_walk}")
print(f"RMSE (Modelo Random Walk): {rmse_random_walk}")

# Modelos de suavização exponencial
y_pred = ses_model(df)
mse_ses = MSE(df['Consumption'], y_pred)
rmse_ses = RMSE(df['Consumption'], y_pred)

y_pred = holt_model(df)
mse_holt = MSE(df['Consumption'], y_pred)
rmse_holt = RMSE(df['Consumption'], y_pred)

y_pred = holt_winters_add_model(df)
mse_hw_add = MSE(df['Consumption'], y_pred)
rmse_hw_add = RMSE(df['Consumption'], y_pred)

constant = abs(df['Consumption'].min()) + 1 
df['Consumption'] = df['Consumption'] + constant
print(df)
y_pred = holt_winters_mul_model(df)
mse_hw_mul = MSE(df['Consumption'], y_pred)
rmse_hw_mul = RMSE(df['Consumption'], y_pred)

print(f"MSE (Modelo SES): {mse_ses}")
print(f"RMSE (Modelo SES): {rmse_ses}")
print(f"MSE (Modelo Holt): {mse_holt}")
print(f"RMSE (Modelo Holt): {rmse_holt}")
print(f"MSE (Modelo Holt-Winters Add): {mse_hw_add}")
print(f"RMSE (Modelo Holt-Winters Add): {rmse_hw_add}")
print(f"MSE (Modelo Holt-Winters Mul): {mse_hw_mul}")
print(f"RMSE (Modelo Holt-Winters Mul): {rmse_hw_mul}")

# Modelo de Regressão Linear Múltipla
y_train, y_test, y_train_pred, y_test_pred = mult_linear_regression(df)
mse_train = MSE(y_train, y_train_pred)
rmse_train = RMSE(y_train, y_train_pred)
mse_test = MSE(y_test, y_test_pred)
rmse_test = RMSE(y_test, y_test_pred)

print(f"MSE (Treino): {mse_train}")
print(f"RMSE (Treino): {rmse_train}")
print(f"MSE (Teste): {mse_test}")
print(f"RMSE (Teste): {rmse_test}")

# Modelo SARIMA
test, y_pred = sarima_model(df)
mse_sarima = MSE(test, y_pred)
rmse_sarima = RMSE(test, y_pred)

print(f"MSE (Modelo SARIMA): {mse_sarima}")
print(f"RMSE (Modelo SARIMA): {rmse_sarima}")

# Modelo de convolução 1D
y_pred, test = conv_1d(df)
mse_conv = MSE(test, y_pred)
rmse_conv = RMSE(test, y_pred)

print(f"MSE (Modelo de Convolução 1D): {mse_conv}")
print(f"RMSE (Modelo de Convolução 1D): {rmse_conv}")