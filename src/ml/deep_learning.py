import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


class ScalerDatos:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def escalar(self, X, y):
        X_escalado = self.scaler_X.fit_transform(X)
        y_escalado = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_escalado, y_escalado

    def desescalar_predicciones(self, predicciones):
        return self.scaler_y.inverse_transform(predicciones.reshape(-1, 1)).flatten()


class RedNeuronalDensa:
    """Red neuronal densa para predicción de demanda (sklearn MLP)."""

    def __init__(self, tamano_entrada: int, unidades_ocultas: list = None,
                 tasa_aprendizaje: float = 0.001, regularizacion_l2: float = 0.01):
        self.tamano_entrada = tamano_entrada
        self.unidades_ocultas = tuple(unidades_ocultas or [128, 64, 32])
        self.tasa_aprendizaje = tasa_aprendizaje
        self.regularizacion_l2 = regularizacion_l2
        self.modelo = None
        self.scaler = ScalerDatos()

    def construir_modelo(self):
        self.modelo = MLPRegressor(
            hidden_layer_sizes=self.unidades_ocultas,
            activation="relu",
            solver="adam",
            learning_rate_init=self.tasa_aprendizaje,
            alpha=self.regularizacion_l2,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            verbose=False,
        )
        return self.modelo

    def entrenar(self, X: pd.DataFrame, y: pd.Series,
                 epocas: int = 100, tamano_lote: int = 32,
                 validacion_split: float = 0.2):
        X_esc, y_esc = self.scaler.escalar(X.values, y.values)
        self.modelo.fit(X_esc, y_esc)
        return self.modelo

    def predecir(self, X: pd.DataFrame):
        X_esc = self.scaler.scaler_X.transform(X.values)
        pred_esc = self.modelo.predict(X_esc)
        return self.scaler.desescalar_predicciones(pred_esc)


class RedLSTM:
    """Aproximación LSTM con MLP sobre ventana temporal."""

    def __init__(self, ventana_temporal: int = 7, unidades_lstm: int = 64,
                 tasa_aprendizaje: float = 0.001):
        self.ventana_temporal = ventana_temporal
        self.unidades_lstm = unidades_lstm
        self.tasa_aprendizaje = tasa_aprendizaje
        self.modelo = None
        self.scaler = MinMaxScaler()
        self.X_validacion = None
        self.y_validacion = None
        self.X_prueba = None
        self.y_prueba = None

    def crear_secuencias(self, datos: np.ndarray, ventana: int):
        X, y = [], []
        for i in range(len(datos) - ventana):
            X.append(datos[i:i + ventana].flatten())
            y.append(datos[i + ventana])
        return np.array(X), np.array(y)

    def preparar_datos(self, demanda: np.ndarray):
        demanda_esc = self.scaler.fit_transform(demanda.reshape(-1, 1))
        X, y = self.crear_secuencias(demanda_esc, self.ventana_temporal)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        self.X_prueba = X_test
        self.y_prueba = y_test
        return X_train, X_test, y_train, y_test

    def construir_modelo(self):
        self.modelo = MLPRegressor(
            hidden_layer_sizes=(self.unidades_lstm, 32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=self.tasa_aprendizaje,
            max_iter=100,
            early_stopping=True,
            random_state=42,
            verbose=False,
        )
        return self.modelo

    def entrenar(self, X_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray,
                 X_validacion: np.ndarray = None, y_validacion: np.ndarray = None,
                 epocas: int = 50, tamano_lote: int = 16):
        self.X_validacion = X_validacion
        self.y_validacion = y_validacion
        self.modelo.fit(X_entrenamiento, y_entrenamiento)
        return self.modelo

    def predecir(self, X):
        pred_esc = self.modelo.predict(X).reshape(-1, 1)
        return self.scaler.inverse_transform(pred_esc)

    def obtener_predicciones_y_reales(self):
        X_eval = self.X_validacion if self.X_validacion is not None else self.X_prueba
        y_eval = self.y_validacion if self.y_validacion is not None else self.y_prueba
        if X_eval is None or y_eval is None:
            raise ValueError("No hay datos de validación/prueba almacenados.")
        pred_esc = self.modelo.predict(X_eval).reshape(-1, 1)
        predicciones = self.scaler.inverse_transform(pred_esc).flatten()
        y_reales = self.scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
        return predicciones, y_reales


class EnsembleRedNeuronal:
    """Ensemble de RedNeuronalDensa + RedLSTM."""

    def __init__(self):
        self.red_densa = None
        self.red_lstm = None

    def entrenar_ensemble(self, X: pd.DataFrame, y: pd.Series,
                          demanda_serie: np.ndarray = None):
        print("Entrenando Red Neuronal Densa...")
        self.red_densa = RedNeuronalDensa(tamano_entrada=X.shape[1])
        self.red_densa.construir_modelo()
        self.red_densa.entrenar(X, y, epocas=100)

        if demanda_serie is not None:
            print("Entrenando Red LSTM (MLP temporal)...")
            self.red_lstm = RedLSTM(ventana_temporal=7)
            self.red_lstm.construir_modelo()
            X_tr, X_te, y_tr, y_te = self.red_lstm.preparar_datos(demanda_serie)
            self.red_lstm.entrenar(X_tr, y_tr, X_te, y_te, epocas=50)

    def predecir_ensemble(self, X: pd.DataFrame, secuencias_lstm: np.ndarray = None,
                          pesos: list = None):
        if pesos is None:
            pesos = [0.5, 0.5] if self.red_lstm else [1.0]

        predicciones = []
        if self.red_densa:
            predicciones.append(self.red_densa.predecir(X) * pesos[0])
        if self.red_lstm and secuencias_lstm is not None:
            peso_lstm = pesos[1] if len(pesos) > 1 else 0.0
            predicciones.append(self.red_lstm.predecir(secuencias_lstm).flatten() * peso_lstm)

        return np.array(predicciones).mean(axis=0) if predicciones else None
