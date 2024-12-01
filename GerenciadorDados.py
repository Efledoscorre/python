import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GerenciadorDados:

    def __init__(self, caminho_arquivo_csv):
        """
        Guarda todas as colunas do csv
        """
        self.caminho_arquivo_csv = caminho_arquivo_csv
        self.data = None
        self.colunas_numericas = [
            'Land Area(Km2)', 'Armed Forces size', 'Birth Rate',
            'Calling Code', 'Capital/Major City', 'Co2-Emissions',
            'CPI', 'CPI Change (%)', 'Currency-Code', 'Fertility Rate',
            'Forested Area (%)', 'Gasoline Price', 'GDP',
            'Gross primary education enrollment (%)',
            'Gross tertiary education enrollment (%)',
            'Infant mortality', 'Largest city', 'Life expectancy',
            'Maternal mortality ratio', 'Minimum wage',
            'Official language', 'Out of pocket health expenditure',
            'Physicians per thousand', 'Population',
            'Population: Labor force participation (%)',
            'Tax revenue (%)', 'Total tax rate', 'Unemployment rate',
            'Urban_population', 'Latitude', 'Longitude'
        ]


    def carregar_dados(self):
        """
        Carregar dados a partir de um arquivo CSV
        """

        self.data = pd.read_csv(self.caminho_arquivo_csv)
        for col in self.colunas_numericas:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')


    def calcular_peso_total(self, pesos):
        """
        Calcular o peso total de cada país com base nos pesos fornecidos
        """
        if not self.data.empty:
            self.data['peso_total'] = self.data[self.colunas_numericas].mul(list(pesos.values()), axis=1).sum(axis=1)
        else:
            print("Erro: Dados não carregados.")


    def mostrar_dados(self):
        """
        Exibir as primeiras linhas dos dados
        """
        print(self.data.head())
