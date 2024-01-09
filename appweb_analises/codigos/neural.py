import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Função para obter as colunas escolhidas pelo usuário
def get_user_columns(df):
    print("Colunas disponíveis: ", df.columns)
    ds_col = input("Escolha a coluna para data (ds): ")
    y_col = input("Escolha a coluna para variável dependente (y): ")
    return ds_col, y_col

def main():
    # Carrega o DataFrame
    df = pd.read_csv('dados_processados.csv')

    # Obtém as colunas escolhidas pelo usuário
    ds_col, y_col = get_user_columns(df)
    df = df[[ds_col, y_col]]
    df[ds_col] = pd.to_datetime(df[ds_col])
    df = df.rename(columns={ds_col: 'ds', y_col: 'y'})

    # Visualiza os dados
    plt.figure(figsize=(10,5))
    plt.plot(df['ds'], df['y'], label='Dados reais')
    plt.title('Impressões ao longo do tempo')
    plt.xlabel('Data')
    plt.ylabel('Impressões')
    plt.legend()
    plt.show()

    # Modelagem com Prophet
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(df)

    # Previsão
    meses_para_prever = int(input("Quantos meses você quer prever? "))
    dias_para_prever = meses_para_prever * 30  # Aproximadamente 30 dias por mês
    future = model.make_future_dataframe(periods=dias_para_prever)
    forecast = model.predict(future)

    # Visualização das previsões
    fig = model.plot(forecast)
    plt.show()

    # Componentes do modelo (opcional)
    fig2 = model.plot_components(forecast)
    plt.show()

if __name__ == '__main__':
    main()
