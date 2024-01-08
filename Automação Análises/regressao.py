import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

def main():
    try:
        # Carrega o DataFrame
        df = pd.read_csv('dados_processados.csv')

        # Obtém as colunas escolhidas pelo usuário
        x_col, y_col = get_user_columns(df)

        # Prepara os dados para o modelo
        X = df[[x_col]]
        y = df[y_col]

        # Divide os dados em conjunto de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Cria e treina o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Faz previsões e avalia o modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Solicita ao usuário para inserir um valor para a variável independente
        while True:
            try:
                valor_independente = float(input(f"Insira o valor para '{x_col}' para prever '{y_col}': "))
                break
            except ValueError:
                print("Entrada inválida. Insira um número válido.")

        # Faz a previsão para esse valor
        valor_previsto = model.predict([[valor_independente]])

        # Exibe o valor previsto
        print(f"Valor previsto para '{y_col}' com base em {valor_independente} '{x_col}': {valor_previsto[0]}")

        # Exibe os resultados
        print("Coeficientes: ", model.coef_)
        print("Intercept: ", model.intercept_)
        print("Mean squared error: ", mse)
        print("Coeficiente de determinação (R^2): ", r2)

    except FileNotFoundError:
        print("Arquivo 'dados_processados.csv' não encontrado.")
    except Exception as e:
        print("Ocorreu um erro:", str(e))

if __name__ == '__main__':
    main()
