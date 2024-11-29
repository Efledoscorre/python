import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd

class ModeloRecomendacao(nn.Module):
    def __init__(self, input_dim):
        """
        Inicializa o modelo e define a arquitetura.
        :param input_dim: Número de entradas (features do dataset).
        """
        super(ModeloRecomendacao, self).__init__()
        self.input_dim = input_dim
        self.modelo = self._criar_modelo()

    def _criar_modelo(self):
        """
        Cria o modelo neural.
        :return: Modelo PyTorch.
        """
        return nn.Sequential(
            nn.Linear(self.input_dim, 64),  # Primeira camada oculta
            nn.ReLU(),
            nn.Linear(64, 32),  # Segunda camada oculta
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16 , 1),
            nn.ReLU(),
            nn.Sigmoid()  # Função de ativação para classificação binária
        )

    def forward(self, x):
        return self.modelo(x)

    def treinar(self, X_train, y_train, epochs=10, batch_size=16, learning_rate=0.001):

        # Converter dados para tensores PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        # Definir função de perda e otimizador
        criterio = nn.BCELoss()  # Binary Cross Entropy Loss para classificação binária
        otimizador = optim.Adam(self.parameters(), lr=learning_rate)

        # Loop de treinamento
        self.train()
        for epoca in range(epochs):
            for i in range(0, len(X_train), batch_size):
                # Dividir os dados em batches
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                previsoes = self.forward(X_batch)
                perda = criterio(previsoes, y_batch)

                # Backward pass e atualização dos pesos
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()

            # Mostrar perda ao final de cada época
            print(f"Época [{epoca + 1}/{epochs}], Perda: {perda.item():.4f}")

    def avaliar(self, X_test, y_test):

        # Converter dados para tensores PyTorch
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Desabilitar gradientes para avaliação
        self.eval()
        with torch.no_grad():
            previsoes = self.forward(X_test)
            previsoes_binarias = (previsoes >= 0.5).float()  # Threshold para binário (>= 0.5 = 1, senão 0)

        # Calcular acurácia
        acuracia = accuracy_score(y_test.numpy(), previsoes_binarias.numpy())
        print(f"Acurácia do modelo: {acuracia:.2f}")

    def recomendar(self, pais_escolhido, dados, colunas_numericas):

        # Verificar se o país existe nos dados
        if pais_escolhido not in dados['Country'].values:
            print(f"País '{pais_escolhido}' não encontrado no dataset.")
            return

        # Obter os dados do país escolhido
        dados_pais = dados[dados['Country'] == pais_escolhido][colunas_numericas].values
        dados_pais_tensor = torch.tensor(dados_pais, dtype=torch.float32)

        # Calcular a distância com todos os países
        self.eval()
        with torch.no_grad():
            distancias = torch.cdist(
                dados_pais_tensor,
                torch.tensor(dados[colunas_numericas].values, dtype=torch.float32)
            )

        # Ordenar por similaridade
        indices_similares = distancias.argsort().numpy()[0][:5]
        recomendacoes = dados.iloc[indices_similares]['Country'].values

        # Mostrar os países recomendados
        print(f"Países similares ao '{pais_escolhido}': {', '.join(recomendacoes)}")

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo (substitua com seus dados reais)
    X_train = [[25, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(100)]
    y_train = [[1] for i in range(100)]
    X_test = [[30, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(10)]
    y_test = [[0] for i in range(10)]




    modelo = ModeloRecomendacao(input_dim=30)
    modelo.treinar(X_train, y_train)
    modelo.avaliar(X_test, y_test)
