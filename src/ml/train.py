import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class EntrenadorDemanda:
    def __init__(self, datos: pd.DataFrame, caracteristicas: list, objetivo: str):
        self.X = datos[caracteristicas]
        self.y = datos[objetivo]
        self.modelos = {
            'RegresionLineal': LinearRegression(),
            'BosqueAleatorio': RandomForestRegressor(n_estimators=100, random_state=42)
        }

    def generar_modelos_entrenados(self) -> tuple:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        modelos_entrenados = {}
        for nombre, modelo in self.modelos.items():
            modelo.fit(X_entrenamiento, y_entrenamiento)
            modelos_entrenados[nombre] = modelo
            
        return modelos_entrenados, X_prueba, y_prueba