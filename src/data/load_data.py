import pandas as pd

class RepositorioVentas:
    def __init__(self, ruta_archivo: str):
        self.ruta_archivo = ruta_archivo

    def cargar_datos_estructurados(self) -> pd.DataFrame:
        datos = pd.read_csv(self.ruta_archivo)
        columnas_requeridas = ['Sales_ID', 'Date_of_Sale', 'Product_Category', 'Sales_Amount']
        return datos[columnas_requeridas]

class AnalizadorVentas:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos

    def obtener_metricas_descriptivas(self) -> pd.DataFrame:
        return self.datos.describe(include='all')