import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


class ModeloRecomendacao(nn.Module):
    """
    Classe que implementa um modelo de rede neural para classificação binária, usando PyTorch.
    """
    def __init__(self, input_dim):
        super(ModeloRecomendacao, self).__init__()
        self.modelo = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        """
        Realiza a passagem para frente dos dados na rede neural.
        """
        return self.modelo(x)


    def treinar(self, X_train, y_train, epochs=10, batch_size=16, learning_rate=0.001):
        """
        Treina o modelo com os dados fornecidos.
        """
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        criterio = nn.BCELoss()
        otimizador = optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoca in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                previsoes = self.forward(X_batch)
                perda = criterio(previsoes, y_batch)

                otimizador.zero_grad()
                perda.backward()
                otimizador.step()

            print(f"Época [{epoca + 1}/{epochs}], Perda: {perda.item():.4f}")


    def avaliar(self, X_test, y_test):
        """
        Avalia o modelo nos dados de teste.
        """
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        self.eval()
        with torch.no_grad():
            previsoes = self.forward(X_test)
            previsoes_binarias = (previsoes >= 0.5).float()

        acuracia = accuracy_score(y_test.numpy(), previsoes_binarias.numpy())
        print(f"Acurácia: {acuracia:.2f}")



def recomendar_paises(modelo, pais_escolhido, dados, colunas_numericas, pesos):
    """
    Faz recomendações de países similares com base no país escolhido.
    """
    pais_info = dados[dados['Country'].str.contains(pais_escolhido, case=False, na=False)]
    if pais_info.empty:
        print(f"País '{pais_escolhido}' não encontrado no dataset.")
        return None

    pais_escolhido = pais_info.iloc[0]['Country']
    dados_pais = pais_info[colunas_numericas].values.flatten()

    dados_pais = pd.Series(dados_pais).fillna(0).values


    dados_pais_ponderados = dados_pais * list(pesos.values())
    similaridades = []
    for index, row in dados.iterrows():
        pais_dados = row[colunas_numericas].values.flatten()
        pais_dados = pd.Series(pais_dados, dtype='float64').fillna(0).values
        pais_dados_ponderados = pais_dados * list(pesos.values())
        similaridade = cosine_similarity([dados_pais_ponderados], [pais_dados_ponderados])[0][0]
        similaridades.append((row['Country'], similaridade))


    similaridades.sort(key=lambda x: x[1], reverse=True)

    recomendacoes = [pais for pais, _ in similaridades[1:6]]

    return recomendacoes