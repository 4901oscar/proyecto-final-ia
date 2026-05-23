import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ProcesadorDemanda:
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos.copy()
        self.codificador = LabelEncoder()

    def ejecutar_transformacion(self) -> pd.DataFrame:
        self.datos['Date_of_Sale'] = pd.to_datetime(self.datos['Date_of_Sale'])

        # Eliminar filas sin monto de venta (dato faltante, no venta cero)
        filas_antes = len(self.datos)
        self.datos = self.datos.dropna(subset=['Sales_Amount'])
        self.nulos_removidos = filas_antes - len(self.datos)

        # Agregar por semana y categoría para reducir ruido diario
        datos_agrupados = (
            self.datos
            .groupby(['Product_Category', pd.Grouper(key='Date_of_Sale', freq='W')])
            ['Sales_Amount'].sum()
            .reset_index()
        )

        datos_agrupados.rename(columns={'Sales_Amount': 'DemandaTotal'}, inplace=True)
        datos_agrupados['Mes'] = datos_agrupados['Date_of_Sale'].dt.month
        datos_agrupados['SemanaDelAno'] = datos_agrupados['Date_of_Sale'].dt.isocalendar().week.astype(int)
        datos_agrupados['Categoria_Codificada'] = self.codificador.fit_transform(datos_agrupados['Product_Category'])

        return datos_agrupados