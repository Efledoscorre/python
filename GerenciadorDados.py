import pandas as pd


class GerenciadorDados:
    def __init__(self, caminho_arquivo):

        self.dados = self.carregar_dados(caminho_arquivo)

    def carregar_dados(self, caminho_arquivo):
        try:

            dados = pd.read_csv(caminho_arquivo)
            return dados
        except FileNotFoundError:
            print(f"Erro: O arquivo {caminho_arquivo} não foi encontrado.")
            return None
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")
            return None

    def carregar_e_preprocessar(self, pesos):

        if not isinstance(self.dados, pd.DataFrame):
            raise TypeError("Os dados devem ser um DataFrame do Pandas.")


        for coluna, peso in pesos.items():
            if coluna not in self.dados.columns:
                print(f"A coluna '{coluna}' não existe nos dados!")
                continue


            self.dados[coluna] = pd.to_numeric(self.dados[coluna], errors='coerce')


            self.dados[coluna] = self.dados[coluna] * peso

    def obter_dados(self):
        return self.dados, self.dados.select_dtypes(include=['number']).columns


