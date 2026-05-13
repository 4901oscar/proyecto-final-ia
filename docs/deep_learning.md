# Deep Learning para Predicción de Demanda

## Introducción

Este documento explica la implementación de **redes neuronales profundas (Deep Learning)** para predicción de demanda en el sistema de retail, usando **TensorFlow/Keras**. El Deep Learning permite capturar patrones complejos y no lineales que los modelos ML tradicionales pueden no detectar.

---

## ¿Qué es Deep Learning?

El Deep Learning es un subcampo del Machine Learning basado en **redes neuronales artificiales** con múltiples capas (de ahí "deep"). Cada capa transforma los datos mediante operaciones matemáticas, permitiendo aprender representaciones cada vez más abstractas.

### Ventajas en Predicción de Demanda

| Aspecto                     | Deep Learning                  | ML Tradicional                    |
| --------------------------- | ------------------------------ | --------------------------------- |
| **Patrones**                | No lineales, muy complejos     | Lineales, moderadamente complejos |
| **Escalabilidad**           | Mejor con muchos datos (>50k)  | Bueno con datos medianos          |
| **Automatización**          | Feature engineering automático | Feature engineering manual        |
| **Tiempo de entrenamiento** | Mayor                          | Menor                             |
| **Interpretabilidad**       | Caja negra                     | Más clara                         |

---

## Arquitecturas Implementadas

### 1. Red Neuronal Densa (Dense Neural Network)

#### Descripción

Una red secuencial con capas completamente conectadas (fully connected). Cada neurona de una capa se conecta con todas las de la siguiente.

#### Arquitectura en el Proyecto

```
Entrada (4 características)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Dense(16) + ReLU + Dropout(0.2)
    ↓
Dense(1) Linear → Predicción de Demanda
```

#### Componentes Clave

- **Dense(n)**: Capa con n neuronas
- **ReLU**: Activación Rectified Linear Unit (introduce no-linealidad)
- **BatchNormalization**: Normaliza la salida de cada capa (estabiliza entrenamiento)
- **Dropout(p)**: Apagra aleatoriamente el p% de neuronas (previene overfitting)
- **L2 Regularization**: Penaliza pesos grandes (evita overfitting)

#### Uso

```python
from src.ml.deep_learning import RedNeuronalDensa

# Crear y entrenar
red_densa = RedNeuronalDensa(tamano_entrada=4)
red_densa.construir_modelo()
historial = red_densa.entrenar(X_entrenamiento, y_entrenamiento, epocas=100)

# Predecir
predicciones = red_densa.predecir(X_nuevos_datos)
```

---

### 2. Red LSTM (Long Short-Term Memory)

#### Descripción

LSTM es una arquitectura especializada para **series temporales**. Puede "recordar" información a largo plazo gracias a su célula de memoria.

#### ¿Por qué LSTM para Demanda?

La demanda típicamente tiene:

- **Tendencias temporales** (estacionalidad, ciclos)
- **Dependencias de largo plazo** (ventas de hoy afectan las de mañana)
- **Patrones cíclicos** (días de la semana, meses)

LSTM captura estas relaciones temporales automáticamente.

#### Arquitectura en el Proyecto

```
Secuencia de 7 días de demanda
    ↓
LSTM(64) con return_sequences=True
    ↓
Dropout(0.2)
    ↓
LSTM(32)
    ↓
Dropout(0.2)
    ↓
Dense(16) + ReLU
    ↓
Dense(1) → Predicción del siguiente día
```

#### Células LSTM

Cada célula LSTM tiene:

- **Cell State (Ct)**: Memoria a largo plazo
- **Hidden State (ht)**: Memoria a corto plazo
- **3 Puertas**:
  - _Forget Gate_: Decide qué olvidar
  - _Input Gate_: Decide qué recordar
  - _Output Gate_: Decide qué pasar al siguiente

#### Uso

```python
from src.ml.deep_learning import RedLSTM

# Crear y entrenar
red_lstm = RedLSTM(ventana_temporal=7)
red_lstm.construir_modelo()

X_train, X_test, y_train, y_test = red_lstm.preparar_datos(demanda_serie)
historial = red_lstm.entrenar(X_train, y_train, X_validacion=X_test, y_validacion=y_test)

# Predecir
predicciones = red_lstm.predecir(X_test)
```

---

## Preprocesamiento de Datos

### Normalización (MinMaxScaler)

```python
# Escalar a [0, 1]
datos_escalados = (datos - min) / (max - min)
```

**Por qué es crítico**: Las redes neuronales convergen mejor cuando los datos están normalizados. Sin normalización, el gradiente puede explotar o desvanecerse.

### Feature Engineering

Para este proyecto:

```python
df['Mes'] = df['Date_of_Sale'].dt.month              # 1-12
df['DiaSemana'] = df['Date_of_Sale'].dt.dayofweek   # 0-6
df['Trimestre'] = df['Date_of_Sale'].dt.quarter     # 1-4
df['Categoria_Codificada'] = LabelEncoder().fit_transform(category)
```

Estos features capturan **patrones temporales** clave en la demanda.

### División de Datos

```
Total: 100,000 registros
├── Entrenamiento: 72% (usado para entrenar)
├── Validación: 8% (usado para ajustar hiperparámetros)
└── Prueba: 20% (usado para evaluar final, nunca en entrenamiento)
```

---

## Hiperparámetros Principales

| Parámetro             | Valor       | Propósito                                 |
| --------------------- | ----------- | ----------------------------------------- |
| **Learning Rate**     | 0.001       | Velocidad de aprendizaje (Adam optimizer) |
| **Batch Size**        | 32          | Muestras por actualización de pesos       |
| **Épocas**            | 100         | Iteraciones máximas                       |
| **Dropout**           | 0.2-0.3     | Prevención de overfitting                 |
| **L2 Regularization** | 0.01        | Penalización de pesos grandes             |
| **Early Stopping**    | patience=15 | Detiene si no mejora validación           |

### Ajuste de Hiperparámetros

Para mejorar resultados:

```python
# Aumentar complejidad del modelo
unidades_ocultas = [256, 128, 64, 32]  # Más capas/neuronas

# Entrenar más
epocas = 200
tamano_lote = 16

# Cambiar learning rate
tasa_aprendizaje = 0.0005
```

---

## Callbacks (Mecanismos de Control)

### Early Stopping

Detiene el entrenamiento si la pérdida de validación no mejora por N épocas.

```python
EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
```

### ReduceLROnPlateau

Reduce la tasa de aprendizaje si el modelo se estanca.

```python
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
```

**Beneficio**: Converge mejor y evita mínimos locales.

---

## Métricas de Evaluación

### Para Regresión (Predicción de Demanda)

1. **MSE (Mean Squared Error)**

   ```
   MSE = (1/n) * Σ(real - predicción)²
   ```

   - Penaliza errores grandes
   - En escala de [0, ∞)

2. **MAE (Mean Absolute Error)**

   ```
   MAE = (1/n) * Σ|real - predicción|
   ```

   - Más interpretable (unidades originales)
   - Robusto a outliers

3. **RMSE (Root Mean Squared Error)**

   ```
   RMSE = √MSE
   ```

   - Vuelve MSE a escala original
   - Métrica principal

4. **R² Score**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```

   - Rango [0, 1]
   - R² = 1: predicción perfecta
   - R² = 0: predicción = media

---

## Flujo Completo

### Entrenamiento

```python
from src.ml.train import EntrenadorDemanda
from src.ml.evaluate import EvaluadorModelos

# 1. Cargar y preparar datos
datos = pd.read_csv('Retail_Sales_Data.csv')
características = ['Mes', 'DiaSemana', 'Trimestre', 'Categoria_Codificada']

# 2. Entrenar modelos (incluir Deep Learning)
entrenador = EntrenadorDemanda(datos, características, 'DemandaTotal')
modelos, X_prueba, y_prueba = entrenador.generar_modelos_entrenados(incluir_dl=True)

# 3. Evaluar
evaluador = EvaluadorModelos(modelos, X_prueba, y_prueba)
metricas = evaluador.calcular_metricas()
print(evaluador.generar_reporte())
```

### Predicción

```python
# Con Red Neuronal Densa
predicción = modelos['RedNeuronalDensa'].predecir(X_nuevos_datos)

# Con LSTM
predicción = modelos['RedLSTM'].predecir(X_nuevos_datos_series)
```

---

## Comparación: Deep Learning vs ML Tradicional

### Red Neuronal Densa vs RandomForest

| Aspecto                     | Densa NN               | RandomForest |
| --------------------------- | ---------------------- | ------------ |
| **Tiempo entrenamiento**    | Moderado (GPU acelera) | Rápido       |
| **Ajuste fino**             | Muchos hiperparámetros | Pocos        |
| **Interpretabilidad**       | Baja                   | Alta         |
| **Manejo de no-linealidad** | Excelente              | Bueno        |
| **Datos requeridos**        | Muchos (>10k)          | Menos        |

### LSTM vs Regresión Lineal

| Aspecto                          | LSTM                    | Lineal            |
| -------------------------------- | ----------------------- | ----------------- |
| **Patrones temporales**          | Captura automáticamente | No captura        |
| **Estacionalidad**               | Sí                      | Requiere features |
| **Complejidad**                  | Alta                    | Baja              |
| **Predicciones múltiples pasos** | Sí                      | Limitado          |

---

## Troubleshooting Común

### 1. Overfitting (Pérdida de entrenamiento ↓, validación ↑)

**Soluciones**:

```python
# Aumentar Dropout
layers.Dropout(0.5)

# Aumentar L2 Regularization
kernel_regularizer=keras.regularizers.l2(0.1)

# Usar Early Stopping
EarlyStopping(patience=10)

# Menos parámetros
unidades_ocultas = [64, 32]  # Reducir
```

### 2. Underfitting (Ambas pérdidas altas)

**Soluciones**:

```python
# Aumentar complejidad
unidades_ocultas = [256, 128, 64]

# Más épocas
epocas = 200

# Menor Dropout
layers.Dropout(0.1)
```

### 3. Pérdida NaN

**Causas**:

- Datos no normalizados
- Learning rate muy alto
- Valores infinitos en datos

**Soluciones**:

```python
# Normalizar datos
scaler = MinMaxScaler()
datos = scaler.fit_transform(datos)

# Reducir learning rate
optimizer = Adam(learning_rate=0.0001)
```

---

## Recursos y Referencias

- **TensorFlow/Keras Docs**: https://www.tensorflow.org/guide
- **LSTM Explicado**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Best Practices**: Deep Learning by Goodfellow, Bengio & Courville

---

## Próximos Pasos

1. **Ajustar hiperparámetros** basado en métricas de validación
2. **Combinar predicciones** de múltiples modelos (ensemble)
3. **Integrar con CSP** del agente para optimización de rutas
4. **Monitoreo en producción** (reentrenamiento periódico)
5. **Explicabilidad** (usar SHAP o atención para interpretar decisiones)

---

_Documento generado para el Proyecto Final IA - Optimización Inteligente de Inventario y Rutas_
