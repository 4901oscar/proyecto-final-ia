import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ScalerDatos:
    """Escalador de datos para redes neuronales"""
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def escalar(self, X, y):
        X_escalado = self.scaler_X.fit_transform(X)
        y_escalado = self.scaler_y.fit_transform(y.reshape(-1, 1))
        return X_escalado, y_escalado.flatten()
    
    def desescalar_predicciones(self, predicciones):
        return self.scaler_y.inverse_transform(predicciones.reshape(-1, 1)).flatten()


class RedNeuronalDensa:
    """Red neuronal densa para predicción de demanda"""
    def __init__(self, tamano_entrada: int, unidades_ocultas: list = None, 
                 tasa_aprendizaje: float = 0.001, regularizacion_l2: float = 0.01):
        self.tamano_entrada = tamano_entrada
        self.unidades_ocultas = unidades_ocultas or [128, 64, 32]
        self.tasa_aprendizaje = tasa_aprendizaje
        self.regularizacion_l2 = regularizacion_l2
        self.modelo = None
        self.scaler = ScalerDatos()
        
    def construir_modelo(self):
        """Construye la arquitectura de la red neuronal"""
        modelo = models.Sequential()
        
        # Capa de entrada
        modelo.add(layers.Dense(
            self.unidades_ocultas[0],
            activation='relu',
            input_shape=(self.tamano_entrada,),
            kernel_regularizer=keras.regularizers.l2(self.regularizacion_l2)
        ))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Dropout(0.3))
        
        # Capas ocultas
        for unidades in self.unidades_ocultas[1:]:
            modelo.add(layers.Dense(
                unidades,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.regularizacion_l2)
            ))
            modelo.add(layers.BatchNormalization())
            modelo.add(layers.Dropout(0.3))
        
        # Capa de salida
        modelo.add(layers.Dense(1, activation='linear'))
        
        # Compilar
        modelo.compile(
            optimizer=Adam(learning_rate=self.tasa_aprendizaje),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.modelo = modelo
        return modelo
    
    def entrenar(self, X: pd.DataFrame, y: pd.Series, 
                 epocas: int = 100, tamano_lote: int = 32, 
                 validacion_split: float = 0.2):
        """Entrena la red neuronal"""
        # Escalar datos
        X_escalado, y_escalado = self.scaler.escalar(X.values, y.values)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Entrenar
        historial = self.modelo.fit(
            X_escalado, y_escalado,
            epochs=epocas,
            batch_size=tamano_lote,
            validation_split=validacion_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return historial
    
    def predecir(self, X: pd.DataFrame):
        """Realiza predicciones"""
        X_escalado = self.scaler.scaler_X.transform(X.values)
        predicciones_escaladas = self.modelo.predict(X_escalado)
        return self.scaler.desescalar_predicciones(predicciones_escaladas)


class RedLSTM:
    """Red LSTM para predicción de series temporales de demanda"""
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
        """Crea secuencias para LSTM"""
        X, y = [], []
        for i in range(len(datos) - ventana):
            X.append(datos[i:i + ventana])
            y.append(datos[i + ventana])
        return np.array(X), np.array(y)
    
    def preparar_datos(self, demanda: np.ndarray):
        """Prepara datos para LSTM"""
        # Escalar
        demanda_escalada = self.scaler.fit_transform(demanda.reshape(-1, 1))
        
        # Crear secuencias
        X, y = self.crear_secuencias(demanda_escalada, self.ventana_temporal)
        
        # Dividir en entrenamiento y prueba
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        self.X_prueba = X_prueba
        self.y_prueba = y_prueba
        
        return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba
    
    def construir_modelo(self):
        """Construye la arquitectura LSTM"""
        modelo = models.Sequential([
            layers.LSTM(self.unidades_lstm, activation='relu', 
                       input_shape=(self.ventana_temporal, 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        modelo.compile(
            optimizer=Adam(learning_rate=self.tasa_aprendizaje),
            loss='mse',
            metrics=['mae']
        )
        
        self.modelo = modelo
        return modelo
    
    def entrenar(self, X_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray,
                X_validacion: np.ndarray = None, y_validacion: np.ndarray = None,
                epocas: int = 50, tamano_lote: int = 16):
        """Entrena el modelo LSTM"""
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_validacion is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        historial = self.modelo.fit(
            X_entrenamiento, y_entrenamiento,
            epochs=epocas,
            batch_size=tamano_lote,
            validation_data=(X_validacion, y_validacion) if X_validacion is not None else None,
            callbacks=[early_stopping],
            verbose=1
        )

        self.X_validacion = X_validacion
        self.y_validacion = y_validacion
        
        return historial
    
    def predecir(self, X):
        """Realiza predicciones con LSTM"""
        predicciones_escaladas = self.modelo.predict(X)
        return self.scaler.inverse_transform(predicciones_escaladas)

    def obtener_predicciones_y_reales(self):
        """Obtiene predicciones y valores reales en escala original para evaluación."""
        X_eval = self.X_validacion if self.X_validacion is not None else self.X_prueba
        y_eval = self.y_validacion if self.y_validacion is not None else self.y_prueba

        if X_eval is None or y_eval is None:
            raise ValueError("No hay datos de validación/prueba almacenados para evaluar LSTM.")

        predicciones_escaladas = self.modelo.predict(X_eval)
        predicciones = self.scaler.inverse_transform(predicciones_escaladas).flatten()
        y_reales = self.scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
        return predicciones, y_reales


class EnsembleRedNeuronal:
    """Combina múltiples redes neuronales para mejor rendimiento"""
    def __init__(self):
        self.red_densa = None
        self.red_lstm = None
        self.modelos_entrenados = []
    
    def entrenar_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                         demanda_serie: np.ndarray = None):
        """Entrena el ensemble de redes"""
        print("Entrenando Red Neuronal Densa...")
        self.red_densa = RedNeuronalDensa(tamano_entrada=X.shape[1])
        self.red_densa.construir_modelo()
        self.red_densa.entrenar(X, y, epocas=100)
        
        if demanda_serie is not None:
            print("\nEntrenando Red LSTM...")
            self.red_lstm = RedLSTM(ventana_temporal=7)
            self.red_lstm.construir_modelo()
            X_train, X_test, y_train, y_test = self.red_lstm.preparar_datos(demanda_serie)
            self.red_lstm.entrenar(X_train, y_train, X_test, y_test, epocas=50)
    
    def predecir_ensemble(self, X: pd.DataFrame, secuencias_lstm: np.ndarray = None, pesos: list = None):
        """Realiza predicciones con el ensemble"""
        if pesos is None:
            pesos = [0.5, 0.5] if self.red_lstm else [1.0]
        
        predicciones = []
        
        if self.red_densa:
            pred_densa = self.red_densa.predecir(X)
            predicciones.append(pred_densa * pesos[0])

        if self.red_lstm and secuencias_lstm is not None:
            pred_lstm = self.red_lstm.predecir(secuencias_lstm).flatten()
            peso_lstm = pesos[1] if len(pesos) > 1 else 0.0
            predicciones.append(pred_lstm * peso_lstm)
        
        return np.array(predicciones).mean(axis=0) if predicciones else None
