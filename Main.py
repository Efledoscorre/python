
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from GerenciadorDados import GerenciadorDados
from ModeloRecomendacao import ModeloRecomendacao, recomendar_paises
from PaisInfo import PaisInfo




def main():
    caminho_arquivo_csv = 'C:\\Users\\lucas\\PycharmProjects\\pythonProject1\\world-data-2023.csv'  # Defina o caminho correto para o arquivo CSV
    gerenciador = GerenciadorDados(caminho_arquivo_csv)
    gerenciador.carregar_dados()

    # Criando instância da classe PaisInfo
    pais_info = PaisInfo(caminho_arquivo_csv)

    # Pesos ajustados para as colunas
    pesos = {
        'Land Area(Km2)': 0.1,
        'Armed Forces size': 0.2,
        'Birth Rate': 0.05,
        'Calling Code': 0.1,
        'Capital/Major City': 0.05,
        'Co2-Emissions': 0.1,
        'CPI': 0.05,
        'CPI Change (%)': 0.05,
        'Currency-Code': 0.05,
        'Fertility Rate': 0.05,
        'Forested Area (%)': 0.05,
        'Gasoline Price': 0.05,
        'GDP': 0.1,
        'Gross primary education enrollment (%)': 0.1,
        'Gross tertiary education enrollment (%)': 0.05,
        'Infant mortality': 0.1,
        'Largest city': 0.1,
        'Life expectancy': 0.1,
        'Maternal mortality ratio': 0.05,
        'Minimum wage': 0.05,
        'Official language': 0.05,
        'Out of pocket health expenditure': 0.05,
        'Physicians per thousand': 0.05,
        'Population': 0.1,
        'Population: Labor force participation (%)': 0.1,
        'Tax revenue (%)': 0.1,
        'Total tax rate': 0.1,
        'Unemployment rate': 0.1,
        'Urban_population': 0.05,
        'Latitude': 0.1,
        'Longitude': 0.1
    }

    # Calcular o peso total
    gerenciador.calcular_peso_total(pesos)

    # Mostrar os dados com o peso total calculado
    gerenciador.mostrar_dados()

    # Preparar os dados para treinamento
    X = gerenciador.data[gerenciador.colunas_numericas].fillna(0).values
    y = gerenciador.data['peso_total'].values

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalizar os valores do alvo para classificação binária (exemplo: "alta expectativa" = 1, "baixa" = 0)
    y = (y >= y.mean()).astype(int)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    modelo = ModeloRecomendacao(X_train.shape[1])
    modelo.treinar(X_train, y_train, epochs=100)

    # Avaliar o modelo
    modelo.avaliar(X_test, y_test)

    # Entrada do usuário para o país escolhido
    pais_escolhido = input("Digite o nome do país para ver as recomendações: ")

    # Obter as recomendações de países
    recomendacoes = recomendar_paises(modelo, pais_escolhido, gerenciador.data, gerenciador.colunas_numericas, pesos)

    # Exibir as recomendações, se houver
    if recomendacoes is not None:
        print(f"Países recomendados para '{pais_escolhido}': {', '.join(recomendacoes)}")
    else:
        print("Nenhuma recomendação disponível.")

    # Exibir as informações sobre o país escolhido
    latitude, longitude = pais_info.mostrar_pais(pais_escolhido)
    if latitude is not None and longitude is not None:
        print(f"\nCoordenadas de {pais_escolhido}: Latitude {latitude}, Longitude {longitude}")

if __name__ == "__main__":
    main()
