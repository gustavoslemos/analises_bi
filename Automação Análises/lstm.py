import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanSquaredError
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Carregando e preparando os dados
df = pd.read_csv('dados_processados.csv')
df['Data'] = pd.to_datetime(df['Data'])
df.sort_values('Data', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
df[['Impressions', 'Link Clicks']] = scaler.fit_transform(df[['Impressions', 'Link Clicks']])

# Função para criar o dataset
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X, y = create_dataset(df[['Impressions']], df['Impressions'], time_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Função objetivo para otimização Bayesiana
def objective(params):
    model = Sequential()
    model.add(LSTM(int(params['units']), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)

    mse = MeanSquaredError()
    mse.update_state(y_test, model.predict(X_test))
    loss = mse.result().numpy()
    return {'loss': loss, 'status': STATUS_OK}

# Espaço de hiperparâmetros
space = {
    'units': hp.quniform('units', 20, 100, 5),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'epochs': hp.quniform('epochs', 10, 50, 5)
}

# Executando a otimização
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

print("Melhores hiperparâmetros encontrados: ", best)

# Reconstruindo o modelo com os melhores hiperparâmetros
model = Sequential()
model.add(LSTM(int(best['units']), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=int(best['epochs']), batch_size=int(best['batch_size']), verbose=1)

def make_prediction(input_data, model, scaler, time_steps):
    # Garantir que os dados de entrada estejam na forma correta
    # Se o scaler foi ajustado em um DataFrame com duas colunas, precisamos transformar os dados de entrada adequadamente
    # Aqui, estou assumindo que a entrada é apenas para 'Impressions', então adiciono uma coluna de zeros para 'Link Clicks'
    input_data_formatted = np.zeros((input_data.shape[0], 2))
    input_data_formatted[:, 0] = input_data[:, 0]  # Colocando 'Impressions' na primeira coluna

    # Normalizar os dados de entrada
    input_data_normalized = scaler.transform(input_data_formatted)

    # Selecionar apenas a coluna normalizada de 'Impressions' para usar como entrada para a previsão
    input_data_normalized = input_data_normalized[:, 0].reshape(-1, 1)

    # Reformatar os dados de entrada para o formato necessário para a LSTM
    X = np.array([input_data_normalized])

    # Fazer a previsão
    prediction = model.predict(X)

    # Para reverter a normalização, precisamos novamente criar um array de duas colunas
    prediction_formatted = np.zeros((prediction.shape[0], 2))
    prediction_formatted[:, 0] = prediction[:, 0]

    # Reverter a normalização
    prediction_reverted = scaler.inverse_transform(prediction_formatted)

    return prediction_reverted[:, 0]  # Retorna apenas a coluna de 'Impressions'


# Exemplo de uso da função de previsão
last_values = df['Impressions'].tail(time_steps).values.reshape(-1, 1)

# Use a função de previsão para fazer a previsão
predicted_impressions = make_prediction(last_values, model, scaler, time_steps)

# Verifique a forma do predicted_impressions e acesse o valor da previsão corretamente
if predicted_impressions.ndim > 1:
    # Se for uma matriz bidimensional, pegue o primeiro elemento da primeira linha
    predicted_value = predicted_impressions[0, 0]
else:
    # Se for um valor escalar ou uma matriz unidimensional
    predicted_value = predicted_impressions[0]

print("Previsão de Impressions:", predicted_value)