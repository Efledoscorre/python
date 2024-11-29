import pandas as pd
from PaisInfo import PaisInfo
from GeoLocalizador import Geolocalizador
from GerenciadorDados import GerenciadorDados
from ModeloRecomendacao import ModeloRecomendacao
from sklearn.model_selection import train_test_split


''''
if __name__ == "__main__":
    dataset_path = 'C:\\Users\\lucas\\PycharmProjects\\pythonProject1\\world-data-2023.csv'
    pais_info = PaisInfo(dataset_path)

    if pais_info.data is not None:
        print("\nColunas disponíveis no dataset:")
        for idx, column in enumerate(pais_info.data.columns):
            print(f"[{idx}] {column}")

        pais_escolhido = input("\nDigite o nome do país para ver as informações: ")
        latitude, longitude = pais_info.mostrar_pais(pais_escolhido)

        if latitude is not None and longitude is not None:
            geolocalizador = Geolocalizador()
            geolocalizador.obter_localizacao(latitude, longitude)




'''


pesos = {
       'GDP': 3.0, 'Tax revenue (%)': 1.5, 'Unemployment rate': 2.0,
        'Population': 1.0, 'Land Area(Km2)': 0.5, 'Fertility Rate': 1.5,
        'Birth Rate': 1.2, 'Gasoline Price': 2.5, 'Infant mortality': 3.0,
        'Life expectancy': 2.8
    }

gerenciador_dados = GerenciadorDados('C:\\Users\\lucas\\PycharmProjects\\pythonProject1\\world-data-2023.csv')
gerenciador_dados.carregar_e_preprocessar(pesos)
dados, colunas_numericas = gerenciador_dados.obter_dados()


X = dados[colunas_numericas].values  # Todas as colunas numéricas
y = dados['Life expectancy'].values  # Coluna alvo (ajuste conforme necessário)

# Normalizar os valores do alvo para classificação binária (exemplo: "alta expectativa" = 1, "baixa" = 0)
# Aqui estou usando uma divisão arbitrária em base de 75 anos
y = (y >= 75).astype(int)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de recomendação
modelo_recomendacao = ModeloRecomendacao(input_dim=len(colunas_numericas))
modelo_recomendacao.treinar(X_train, y_train, epochs=10, batch_size=16)

# Avaliar o modelo
modelo_recomendacao.avaliar(X_test, y_test)
