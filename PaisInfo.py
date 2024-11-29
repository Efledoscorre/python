import pandas as pd

class PaisInfo:
    def __init__(self, dataset_path):
        print("Carregando os dados do CSV...")
        self.data = self.carregar_dados(dataset_path)

    def carregar_dados(self, dataset_path):
        try:
            data = pd.read_csv(dataset_path)

            data.columns = data.columns.str.replace('\n', '', regex=False)
            data.columns = data.columns.str.strip()
            print("Dados carregados com sucesso!")
            return data
        except FileNotFoundError:
            print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
            return None
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")
            return None

    def mostrar_pais(self, pais):
        pais = pais.strip()
        pais_info = self.data[self.data['Country'].str.contains(pais, case=False, na=False)]

        if pais_info.empty:
            print(f"Não encontramos informações sobre o país '{pais}'. Verifique a grafia.")
            return None, None
        else:
            print(f"\nInformações sobre '{pais}':")
            print(pais_info.to_string(index=False))

            latitude = pais_info['Latitude'].values[0]
            longitude = pais_info['Longitude'].values[0]
            return latitude, longitude
