
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from GerenciadorDados import GerenciadorDados
from ModeloRecomendacao import ModeloRecomendacao, recomendar_paises
from PaisInfo import PaisInfo


def main():
    """
    Função principal para executar o sistema de recomendação de países.

    O sistema carrega os dados de um arquivo CSV contendo informações sobre países,
    calcula os pesos ajustados para diversas características, treina um modelo de
    recomendação e, finalmente, fornece recomendações com base no país escolhido pelo
    usuário.
    """


    caminho_arquivo_csv = 'C:\\Users\\lucas\\PycharmProjects\\pythonProject1\\world-data-2023.csv'  # Defina o caminho correto para o arquivo CSV
    gerenciador = GerenciadorDados(caminho_arquivo_csv)
    gerenciador.carregar_dados()


    pais_info = PaisInfo(caminho_arquivo_csv)


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


    gerenciador.calcular_peso_total(pesos)


    gerenciador.mostrar_dados()


    X = gerenciador.data[gerenciador.colunas_numericas].fillna(0).values
    y = gerenciador.data['peso_total'].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    y = (y >= y.mean()).astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    modelo = ModeloRecomendacao(X_train.shape[1])
    modelo.treinar(X_train, y_train, epochs=100)


    modelo.avaliar(X_test, y_test)


    pais_escolhido = input("Digite o nome do país (em inglês) para ver as recomendações: ")


    recomendacoes = recomendar_paises(modelo, pais_escolhido, gerenciador.data, gerenciador.colunas_numericas, pesos)


    if recomendacoes is not None:
        print(f"Países recomendados para '{pais_escolhido}': {', '.join(recomendacoes)}")
    else:
        print("Nenhuma recomendação disponível.")


    latitude, longitude = pais_info.mostrar_pais(pais_escolhido)
    if latitude is not None and longitude is not None:
        print(f"\nCoordenadas de {pais_escolhido}: Latitude {latitude}, Longitude {longitude}")

if __name__ == "__main__":
    main()
