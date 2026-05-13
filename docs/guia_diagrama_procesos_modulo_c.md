# Guia de procesos del modulo C (Deep Learning)

## Objetivo

Este documento explica de principio a fin como opera el modulo C (Deep Learning), cuales son sus entradas y salidas, y como se conecta con los modulos A, B, D y E.

## 1) Entradas del modulo C

El modulo C recibe dos tipos de entradas:

1. Dataset transaccional procesado por el pipeline de datos.
   Campos minimos esperados:

- Date_of_Sale
- Product_Category
- Sales_Amount

2. Features tabulares para red densa (desde preprocessing/modulo B):

- Mes
- DiaSemana
- Categoria_Codificada

3. Serie temporal de demanda (para LSTM):

- Vector unidimensional con DemandaTotal ordenada por tiempo.

## 2) Transformaciones internas del modulo C

Flujo interno implementado:

1. Escalado de datos

- MinMaxScaler para X y y en la red densa.
- MinMaxScaler para la serie en LSTM.

2. Red neuronal densa

- Entrada: matriz tabular (n_muestras, n_features).
- Arquitectura: Dense + BatchNorm + Dropout + Dense final lineal.
- Salida: prediccion de demanda en escala original.

3. Red LSTM

- Entrada: secuencias (n_muestras, ventana_temporal, 1).
- Se crean secuencias con ventana temporal (por defecto 7).
- Split temporal train/test sin barajado para respetar serie.
- Salida: prediccion de demanda en escala original.

4. Evaluacion

- Metricas: MSE, MAE, RMSE, R2.
- La evaluacion de LSTM usa su propio bloque de validacion/prueba almacenado.

5. Ensemble (opcional)

- Combina salida de red densa y LSTM cuando se proporcionan secuencias LSTM en inferencia.

## 3) Salidas del modulo C

1. Modelos entrenados:

- RedNeuronalDensa
- RedLSTM

2. Predicciones:

- Prediccion tabular (red densa).
- Prediccion temporal (LSTM).
- Prediccion combinada (ensemble, opcional).

3. Metricas de desempeno:

- MSE
- MAE
- RMSE
- R2

## 4) Interaccion con modulos A, B, D y E

Segun arquitectura del proyecto:

1. Modulo B -> Modulo C

- B entrega dataset transformado y features de demanda.
- C consume esas features y entrena red densa/LSTM.

2. Modulo C -> Modulo A

- C entrega pronostico de demanda por categoria/periodo.
- A usa esa demanda pronosticada para priorizar rutas de recoleccion con busqueda/CSP.

3. Modulo C -> Modulo D

- C entrega resultados y metricas.
- D genera resumen/reportes en lenguaje natural para stakeholders.

4. Modulo C -> Modulo E

- C entrega modelos, predicciones y metricas para integracion.
- E centraliza pipeline final y revisa sesgos/etica sobre predicciones.

## 5) Diagrama de procesos sugerido (vista general)

Puedes copiar este bloque en Mermaid Live Editor o en Markdown compatible:

```mermaid
flowchart LR
    Data[Dataset publico mayor a 50k] --> B[Modulo B: Pipeline ML y features]
    B --> C[Modulo C: Deep Learning]
    C --> C1[Red Densa]
    C --> C2[LSTM]
    C1 --> Out1[Prediccion demanda tabular]
    C2 --> Out2[Prediccion demanda temporal]
    Out1 --> E[Modulo E: Integracion y etica]
    Out2 --> E
    Out1 --> A[Modulo A: Agente y CSP rutas]
    Out2 --> A
    C --> D[Modulo D: NLP LLM reportes]
    D --> E
    A --> E
```

## 6) Guia paso a paso para construir tu diagrama final

1. Define alcance del diagrama: extremo a extremo (datos -> decision).
2. Dibuja primero los bloques A, B, C, D y E.
3. Agrega entradas de C:

- features tabulares
- serie temporal de demanda

4. Agrega procesos internos de C:

- escalado
- entrenamiento densa
- entrenamiento LSTM
- evaluacion
- ensemble opcional

5. Agrega salidas de C:

- predicciones
- metricas
- modelos entrenados

6. Conecta C con A, D y E segun las dependencias anteriores.
7. Valida con el equipo que cada flecha tenga contrato de datos claro.

## 7) Contratos de datos recomendados para documentar

Para cada flecha del diagrama, agrega:

- Nombre del payload
- Formato (DataFrame, vector, JSON)
- Campos obligatorios
- Frecuencia de actualizacion
- Responsable del modulo origen

Ejemplo rapido C -> A:

- Payload: pronostico_demanda
- Formato: DataFrame
- Campos: fecha, categoria, demanda_predicha
- Uso en A: priorizar picking y plan de ruta

## 8) Referencias de implementacion (codigo actual)

- src/ml/deep_learning.py
- src/ml/train.py
- src/ml/evaluate.py
- docs/arquitectura.txt
- docs/asignacion_modulos.md
