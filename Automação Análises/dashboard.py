import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    # Carrega o DataFrame
    df = pd.read_csv('dados_processados.csv')
    return df

def run_regression(df, x_col, y_col, valor_independente):
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

    # Faz a previsão para o valor independente fornecido
    valor_previsto = model.predict([[valor_independente]])

    return valor_previsto[0], model.coef_, model.intercept_, mse, r2

def main():
    st.title("Dashboard de Análise de Dados com Regressão Linear")

    df = load_data()
    st.write(df.head())

    if not df.empty:
        # Escolha das colunas para regressão
        colunas = df.columns.tolist()
        x_col = st.selectbox("Escolha a coluna para variável independente:", colunas)
        y_col = st.selectbox("Escolha a coluna para variável dependente:", colunas)

        valor_independente = st.number_input(f"Insira o valor para '{x_col}' para prever '{y_col}':", format="%.2f")

        if st.button("Executar Regressão Linear"):
            valor_previsto, coef, intercept, mse, r2 = run_regression(df, x_col, y_col, valor_independente)

            st.write(f"Valor previsto para '{y_col}' com base em {valor_independente} '{x_col}': {valor_previsto}")
            st.write("Coeficientes: ", coef)
            st.write("Intercept: ", intercept)
            st.write("Mean squared error: ", mse)
            st.write("Coeficiente de determinação (R^2): ", r2)

if __name__ == '__main__':
    main()
