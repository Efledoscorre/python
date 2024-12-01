class InterfaceUsuario:
    def __init__(self, data):
        self.data = data


    def listar_colunas(self):
        """
        Lista as colunas disponíveis no dataset.
        """

        print("\nColunas disponíveis no dataset:")
        for idx, column in enumerate(self.data.columns):
            print(f"[{idx}] {column}")


    def exibir_informacoes_pais(self):
        """
        Exibe as informações do país selecionado pelo usuário.
        """

        pais_escolhido = input("\nDigite o nome do país para ver as informações: ")
        if pais_escolhido in self.data['Country'].values:
            print("\nInformações do país escolhido:")
            print(self.data[self.data['Country'] == pais_escolhido].to_string(index=False))
        else:
            print(f"O país '{pais_escolhido}' não foi encontrado no dataset.")
