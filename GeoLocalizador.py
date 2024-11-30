import time
from geopy.geocoders import Nominatim

class Geolocalizador:
    def __init__(self, user_agent="MeuProjeto"):
        self.geolocator = Nominatim(user_agent=user_agent)

    def obter_localizacao(self, lat, lon):
        print(f"Obtendo localização para Latitude: {lat}, Longitude: {lon}")
        time.sleep(1)
        location = self.geolocator.reverse((lat, lon), language='pt', exactly_one=True)

        if location:
            address = location.raw['address']
            pais = address.get('country', 'Desconhecido')
            continente = address.get('continent', 'Desconhecido')
            print(f"País: {pais}, Continente: {continente}")
        else:
            print("Localização não encontrada.")
