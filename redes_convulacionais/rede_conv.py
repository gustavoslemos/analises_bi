import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Importe os dados
df = pd.read_csv("impressões.csv")

# Converta a data para um formato datetime
df['data'] = pd.to_datetime(df['data'])

# Selecione as colunas de data e impressões
df = df[['data', 'impressões']]

# Divida os dados em dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df, df['impressões'], test_size=0.25)

# Transforme os dados de treinamento e teste em uma matriz de imagens
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Crie um loop para ajustar os hiperparâmetros
for num_filters in [32, 64, 128]:
    for kernel_size in [3, 5, 7]:
        for num_layers_pooling in [1, 2, 3]:
            for num_units_dense in [10, 50, 100]:
                for loss in ['mse', 'mae']:
                    for optimizer in ['adam', 'rmsprop']:
                        # Cria o modelo com os novos hiperparâmetros
                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu'),
                            tf.keras.layers.MaxPooling1D(2),
                            # tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu'),
                            # tf.keras.layers.MaxPooling1D(2),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(num_units_dense, activation='relu'),
                            tf.keras.layers.Dense(1)
                        ])

                        # Treina o modelo
                        model.compile(loss=loss, optimizer=optimizer, metrics=['mse', 'mae', 'accuracy', 'precision', 'recall'])
                        model.fit(X_train, y_train, epochs=10)

                        # Avalie o modelo
                        y_pred = model.predict(X_test)

                        # Calcula as métricas
                        mse = np.mean(np.power(y_pred - y_test, 2))
                        mae = np.mean(np.abs(y_pred - y_test))
                        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
                        precision = tf.metrics.precision(y_true=y_test, y_pred=y_pred)[1]
                        recall = tf.metrics.recall(y_true=y_test, y_pred=y_pred)[1]

                        # Imprime os resultados
                        print(f"Número de filtros: {num_filters}, tamanho do kernel: {kernel_size}, número de camadas de pooling: {num_layers_pooling}, número de unidades na camada densa: {num_units_dense}, função de perda: {loss}, otimizador: {optimizer}, MSE: {mse}, MAE: {mae}, Precisão: {accuracy}, Recall: {recall}")
                        print(f"O MSE foi {mse}, no entanto o melhor é 5 ou menos")

# Faça previsões para 3 meses após a última data do dataset
data_futura = pd.to_datetime(df['data'].iloc[-1]) + pd.to_timedelta(days=90)
X_futura = np.array([data_futura])
y_futuro = model.predict(X_futura)

# Plote os valores previstos
plt.plot(df['data'], df['impressões'], label='Dados reais')
plt.plot(data_futura, y_futuro, label='Previsão')
plt
