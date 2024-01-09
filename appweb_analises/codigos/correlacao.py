import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_user_columns(df):
    print("Colunas disponíveis: ", df.columns)
    while True:
        x_col = input("Escolha a coluna para variável independente: ")
        if x_col in df.columns:
            break
        else:
            print("Coluna não encontrada. Escolha uma coluna válida.")

    while True:
        y_col = input("Escolha a coluna para variável dependente: ")
        if y_col in df.columns:
            break
        else:
            print("Coluna não encontrada. Escolha uma coluna válida.")

    return x_col, y_col

def plot_correlation(df, x_col, y_col):
    # Calcula a correlação
    correlation = df[[x_col, y_col]].corr().iloc[0, 1]
    print(f"Correlação entre {x_col} e {y_col}: {correlation}")

    # Cria um gráfico de dispersão
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f"Gráfico de Dispersão entre {x_col} e {y_col}")
    plt.show()

def main():
    try:
        # Carrega o DataFrame
        df = pd.read_csv('dados_processados.csv')

        # Obtém as colunas escolhidas pelo usuário
        x_col, y_col = get_user_columns(df)

        # Plota a correlação
        plot_correlation(df, x_col, y_col)

        # Resto do código de regressão...

    except FileNotFoundError:
        print("Arquivo 'dados_processados.csv' não encontrado.")
    except Exception as e:
        print("Ocorreu um erro:", str(e))

if __name__ == '__main__':
    main()
