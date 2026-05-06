import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ProcesadorDemanda:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos.copy()
        self.codificador = LabelEncoder()

    def ejecutar_transformacion(self) -> pd.DataFrame:
        self.datos['Date_of_Sale'] = pd.to_datetime(self.datos['Date_of_Sale'])
        self.datos['Mes'] = self.datos['Date_of_Sale'].dt.month
        self.datos['DiaSemana'] = self.datos['Date_of_Sale'].dt.dayofweek
        
        datos_agrupados = self.datos.groupby(
            ['Date_of_Sale', 'Product_Category', 'Mes', 'DiaSemana']
        )['Sales_Amount'].sum().reset_index()
        
        datos_agrupados.rename(columns={'Sales_Amount': 'DemandaTotal'}, inplace=True)
        
        datos_agrupados['Categoria_Codificada'] = self.codificador.fit_transform(datos_agrupados['Product_Category'])
        
        return datos_agrupados