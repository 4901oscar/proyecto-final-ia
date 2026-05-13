import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from deep_learning import RedNeuronalDensa, RedLSTM, EnsembleRedNeuronal

class EntrenadorDemanda:
    def __init__(self, datos: pd.DataFrame, caracteristicas: list, objetivo: str):
        self.X = datos[caracteristicas]
        self.y = datos[objetivo]
        self.modelos_ml = {
            'RegresionLineal': LinearRegression(),
            'BosqueAleatorio': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.modelos_dl = {}
        self.X_entrenamiento = None
        self.X_prueba = None
        self.y_entrenamiento = None
        self.y_prueba = None

    def generar_modelos_entrenados(self, incluir_dl: bool = True) -> tuple:
        """Entrena modelos ML y opcionalmente DL"""
        # División estándar
        self.X_entrenamiento, self.X_prueba, self.y_entrenamiento, self.y_prueba = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        modelos_entrenados = {}
        
        # Entrenar modelos ML
        for nombre, modelo in self.modelos_ml.items():
            modelo.fit(self.X_entrenamiento, self.y_entrenamiento)
            modelos_entrenados[nombre] = modelo
        
        # Entrenar modelos DL si se solicita
        if incluir_dl:
            print("\n=== Entrenando Modelos de Deep Learning ===")
            
            # Red Neuronal Densa
            print("\nConstructor: Red Neuronal Densa...")
            red_densa = RedNeuronalDensa(tamano_entrada=self.X_entrenamiento.shape[1])
            red_densa.construir_modelo()
            red_densa.entrenar(
                self.X_entrenamiento, 
                self.y_entrenamiento,
                epocas=100,
                tamano_lote=32
            )
            self.modelos_dl['RedNeuronalDensa'] = red_densa
            modelos_entrenados['RedNeuronalDensa'] = red_densa
            
            # LSTM para series temporales
            print("\nConstructor: Red LSTM...")
            red_lstm = RedLSTM(ventana_temporal=7, unidades_lstm=64)
            red_lstm.construir_modelo()
            
            # Preparar datos para LSTM
            demanda_total = self.y.values
            X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = red_lstm.preparar_datos(demanda_total)
            
            red_lstm.entrenar(
                X_lstm_train,
                y_lstm_train,
                X_validacion=X_lstm_test,
                y_validacion=y_lstm_test,
                epocas=50
            )
            self.modelos_dl['RedLSTM'] = red_lstm
            modelos_entrenados['RedLSTM'] = red_lstm
            
        return modelos_entrenados, self.X_prueba, self.y_prueba
    
    def obtener_predicciones_dl(self, modelos_dl: dict, secuencias_lstm=None):
        """Obtiene predicciones de modelos DL"""
        predicciones = {}
        
        if 'RedNeuronalDensa' in modelos_dl:
            predicciones['RedNeuronalDensa'] = modelos_dl['RedNeuronalDensa'].predecir(self.X_prueba)

        if 'RedLSTM' in modelos_dl and secuencias_lstm is not None:
            predicciones['RedLSTM'] = modelos_dl['RedLSTM'].predecir(secuencias_lstm).flatten()
        
        return predicciones