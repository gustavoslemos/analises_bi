import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from google.cloud import bigquery
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Funções para carregar dados
def load_data_from_bigquery():
    query = "SUA_QUERY_AQUI"
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    return df

def load_data_from_csv():
    file_path = input("Digite o caminho do arquivo CSV: ")
    df = pd.read_csv(file_path)
    return df

def load_data_from_gsheet(url, page_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(url).worksheet(page_name)
    data = sheet.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    return df

def load_data_from_database(tipo, url=None, page=None):
    try:
        if tipo == "gsheets":
            df = load_data_from_gsheet(url, page)
        elif tipo == "bd":
            df = load_data_from_bigquery()
        elif tipo == "csv":
            df = load_data_from_csv()
        else:
            print("Tipo de dataset não reconhecido.")
            exit()

        df.to_csv("temp_data.csv", index=False)
        return df

    except Exception as e:
        print(f"Erro: {e}")
        raise 

def get_user_input():
    tipo_de_dataset = input("Insira o tipo de dataset que irá utilizar(bd,gsheets,csv): ")

    if tipo_de_dataset == "gsheets":
        SHEET_URL = input("Digite o link da planilha Google: ")
        SHEET_PAGE = input("Digite o nome da página da planilha Google: ")
        return tipo_de_dataset, SHEET_URL, SHEET_PAGE

    elif tipo_de_dataset in ["bd", "csv"]:
        return tipo_de_dataset, None, None

    else:
        print("Tipo de dataset não reconhecido.")
        exit()

def process_dataframe(df):
    # Converte as colunas 'Impressions' e 'Link Clicks' para numéricas
    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce').astype(float)
    df['Link Clicks'] = pd.to_numeric(df['Link Clicks'], errors='coerce').astype(float)
    df['Amount Spent'] = df['Amount Spent'].str.replace(',', '.').astype(float)

    # Substitui os valores nulos pela média da coluna 'Amount Spent'
    df['Amount Spent'].fillna(df['Amount Spent'].mean(), inplace=True)


    # Substitui os valores nulos pela média das colunas
    df['Impressions'].fillna(df['Impressions'].mean(), inplace=True)
    df['Link Clicks'].fillna(df['Link Clicks'].mean(), inplace=True)
    # df['Website Leads'].fillna(df['Website Leads'].mean(), inplace=True)
    # df['On-Facebook Leads'].fillna(df['On-Facebook Leads'].mean(), inplace=True)


    # Converte a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    return df

if __name__ == "__main__":
    tipo, url, page = get_user_input()
    df = load_data_from_database(tipo, url, page)
    df = process_dataframe(df)  # Processa o dataframe

    # Salva o dataframe processado em um arquivo CSV
    df.to_csv('dados_processados.csv', index=False)

    print(df.head())
    # Exibe o tipo de dados de cada coluna
    print("\nTipos de Dados das Colunas:")
    print(df.dtypes)
