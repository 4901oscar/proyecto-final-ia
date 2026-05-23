# Documentación Técnica — Módulo B: Pipeline de Machine Learning
**Proyecto:** Sistema de Optimización Inteligente de Inventario y Rutas de Despacho  
**Módulo:** B — Predicción de Demanda  
**Responsable:** José Avila  
**Unidades del curso:** 5 y 6  

---

## 1. Propósito del Módulo

El Módulo B es el núcleo predictivo del sistema. Su responsabilidad es transformar las transacciones históricas de ventas en predicciones de demanda futura por categoría de producto.

**Salidas directas del Módulo B:**
- **→ Módulo C (Deep Learning):** conexión directa en código. `train.py` importa y entrena las redes del Módulo C sobre los mismos datos preprocesados por B.
- **→ Módulo E (Integración):** entrega el modelo entrenado y el diccionario de predicciones por categoría. Es responsabilidad del Módulo E (pipeline.py) pasar ese output al Módulo A.

**Módulo B NO es responsable de** la conexión con Módulo A — esa orquestación pertenece al Módulo E.

---

## 2. Flujo de Datos Completo

```
Retail_Sales_Data.csv  (100,000 transacciones brutas)
         │
         ▼
┌─────────────────────────────────────────────┐
│  src/data/load_data.py                       │
│  RepositorioVentas.cargar_datos_estructurados│
│  → Selecciona columnas relevantes            │
│  → Salida: DataFrame 90,000 filas × 4 cols   │
│    [Sales_ID, Date_of_Sale,                  │
│     Product_Category, Sales_Amount]          │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  src/ml/preprocess.py                        │
│  ProcesadorDemanda.ejecutar_transformacion   │
│                                              │
│  1. Elimina 10,000 filas con                 │
│     Sales_Amount = NaN                       │
│  2. Extrae Mes y SemanaDelAño de la fecha    │
│  3. Agrega por semana × categoría            │
│     (DemandaTotal = sum de Sales_Amount)     │
│  4. Codifica categorías con LabelEncoder     │
│                                              │
│  → Salida: DataFrame 888 filas × 6 cols      │
│    [Product_Category, Date_of_Sale,          │
│     DemandaTotal, Mes, SemanaDelAño,         │
│     Categoria_Codificada]                    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  src/ml/train.py                             │
│  EntrenadorDemanda                           │
│                                              │
│  Features (X): Mes, SemanaDelAño,            │
│                Categoria_Codificada          │
│  Target  (y): DemandaTotal                   │
│                                              │
│  ┌──────────────────────────────────┐        │
│  │  train_test_split 80/20          │        │
│  │  710 entrenamiento / 178 prueba  │        │
│  └──────────────────────────────────┘        │
│                                              │
│  Modelo 1: RegresionLineal (baseline)        │
│  Modelo 2: BosqueAleatorio (100 árboles)     │
│                                              │
│  Método adicional: validar_cruzado(k=5)      │
│  → KFold 5-fold sobre dataset completo       │
│                                              │
│  → Salida: dict{nombre: modelo_entrenado}    │
│            X_prueba, y_prueba                │
│  → Conexión C: incluir_dl=True entrena       │
│    RedNeuronalDensa + RedLSTM sobre mismos   │
│    features                                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  src/ml/evaluate.py                          │
│  EvaluadorModelos.calcular_metricas          │
│                                              │
│  → MAE, MSE, RMSE, R² por modelo            │
│  → Selección del mejor por R²                │
│  → Predicciones por categoría para           │
│    alimentar al Módulo A                     │
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
  Módulo A (Opcional*)       Módulo C (DL)
  Top-3 categorías           Mismo X/y para
  → rutas óptimas            entrenar redes
```

---

## 3. Descripción de Archivos

### `src/data/load_data.py`

```python
class RepositorioVentas:
    columnas_requeridas = ['Sales_ID', 'Date_of_Sale',
                           'Product_Category', 'Sales_Amount']
```

**Decisión:** Se cargan solo las columnas necesarias para el pipeline de ML. Columnas como `Customer_Gender`, `Customer_Age`, `Discount` y `Sales_Region` no se incluyen porque:
- `Discount` tiene r = -0.026 con `DemandaTotal` (correlación despreciable, validada empíricamente).
- `Customer_Gender` / `Customer_Age` tienen 10,000 nulos (10%) y agregarlos como features requeriría imputación que podría introducir sesgo (documentado en `ethics_analysis.md`).
- `Sales_Region` y `Sales_Representative` son variables de identificación, no temporales.

---

### `src/ml/preprocess.py`

#### Decisión 1 — Manejo de valores nulos

```python
filas_antes = len(self.datos)
self.datos = self.datos.dropna(subset=['Sales_Amount'])
self.nulos_removidos = filas_antes - len(self.datos)  # = 10,000
```

Las 10,000 filas con `Sales_Amount = NaN` se **eliminan** (no se imputan con 0). Justificación: una transacción sin monto registrado no equivale a una venta de $0 — es un dato faltante. Imputar con 0 inflaría artificialmente los días/semanas de baja demanda y sesgaría el modelo hacia abajo.

#### Decisión 2 — Agregación semanal en lugar de diaria

```python
self.datos.groupby(
    ['Product_Category', pd.Grouper(key='Date_of_Sale', freq='W')]
)['Sales_Amount'].sum()
```

**Por qué semanal y no diario:**

| Nivel | Muestras | Coeficiente de Variación (Electronics) | R² RF |
|-------|----------|----------------------------------------|-------|
| Diario | 6,096 | 0.290 | -0.27 |
| **Semanal** | **888** | **0.151** | **+0.20** |
| Mensual | ~216 | 0.229 | insuficiente |

La demanda diaria por categoría tiene ruido aleatorio alto (CV=0.29). Agregar por semana reduce el ruido a la mitad (CV=0.15) y permite que el modelo capture patrones estacionales reales. El nivel mensual tiene muy pocas muestras (9 meses × 24 categorías = 216 filas).

#### Decisión 3 — Encoding de categorías

```python
datos_agrupados['Categoria_Codificada'] = \
    self.codificador.fit_transform(datos_agrupados['Product_Category'])
```

Se usa `LabelEncoder` que asigna enteros ordinales (0–23) a las 24 categorías.

**Limitación conocida:** Para `RegresionLineal`, el encoding ordinal introduce una ordenación implícita falsa (como si "Toys=23" fuera "más" que "Books=2"). Para `BosqueAleatorio` esto no tiene impacto porque los árboles hacen splits binarios y no asumen linealidad. La solución correcta para el modelo lineal sería `OneHotEncoder`, pero aumentaría la dimensionalidad de 3 a 26 features y haría el modelo menos interpretable para el dominio.

#### Decisión 4 — Sin normalización de features

No se aplica `StandardScaler` ni `MinMaxScaler`. Justificación:
- `BosqueAleatorio` es invariante a la escala (los árboles solo comparan valores relativos).
- `RegresionLineal` sí se beneficia de normalización para interpretar coeficientes, pero dado que es el modelo baseline y no el modelo principal, se acepta este trade-off a cambio de mantener la interpretabilidad directa de los features.

---

### `src/ml/train.py`

#### Modelos implementados

**Modelo 1 — Regresión Lineal (baseline)**

```python
LinearRegression()
```

- Hipótesis: existe relación lineal entre `{Mes, SemanaDelAño, Categoria_Codificada}` y `DemandaTotal`.
- Ventaja: alta explicabilidad, coeficientes interpretables directamente.
- Desventaja: no captura interacciones no lineales entre features (ej. categoría × estacionalidad).

**Modelo 2 — Bosque Aleatorio (modelo principal)**

```python
RandomForestRegressor(n_estimators=100, random_state=42)
```

- 100 árboles de decisión entrenados en subconjuntos aleatorios del dataset.
- Captura relaciones no lineales y la interacción entre `Categoria_Codificada` y `SemanaDelAño`.
- Robusto a outliers por su naturaleza de ensemble.
- `random_state=42` garantiza reproducibilidad.

#### Validación cruzada 5-fold

```python
def validar_cruzado(self, k: int = 5) -> dict:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores_r2  = cross_val_score(modelo, self.X, self.y, cv=kf, scoring='r2')
    scores_mae = cross_val_score(modelo, self.X, self.y, cv=kf, scoring='neg_mean_absolute_error')
```

La validación cruzada evalúa los modelos sobre 5 particiones distintas del dataset, dando un R² promedio más estable que el de un solo split. Esto elimina la dependencia del resultado en la partición aleatoria (`random_state=42`).

#### Conexión con Módulo C

```python
def generar_modelos_entrenados(self, incluir_dl: bool = True):
    ...
    if incluir_dl and _DL_DISPONIBLE:
        red_densa = RedNeuronalDensa(tamano_entrada=self.X_entrenamiento.shape[1])
        red_densa.entrenar(self.X_entrenamiento, self.y_entrenamiento)
```

Cuando `incluir_dl=True`, el mismo `X_entrenamiento` e `y_entrenamiento` que usa Módulo B se pasan directamente a las redes neuronales del Módulo C (`RedNeuronalDensa`, `RedLSTM`). El Módulo B actúa como capa de preparación de datos para ambos paradigmas (ML clásico y Deep Learning).

---

### `src/ml/evaluate.py`

```python
class EvaluadorModelos:
    def calcular_metricas(self) -> dict:
        # Detecta automáticamente si el modelo es ML (.predict) o DL (.predecir)
```

El evaluador es genérico y compatible con modelos ML y DL. Calcula:

| Métrica | Fórmula | Interpretación en el dominio |
|---------|---------|------------------------------|
| **MAE** | mean(\|y - ŷ\|) | Error promedio en quetzales/dólares de ventas semanales |
| **MSE** | mean((y - ŷ)²) | Penaliza errores grandes (desabastecimiento crítico) |
| **RMSE** | √MSE | En las mismas unidades que MAE, más sensible a outliers |
| **R²** | 1 - SS_res/SS_tot | % de varianza de demanda explicada por el modelo |

**Nota metodológica — Por qué no se usan curvas ROC:**  
Las curvas ROC, precisión, recall y F1 aplican exclusivamente a problemas de **clasificación** (salida discreta). Este problema predice `DemandaTotal` (valor continuo), por lo tanto es **regresión**. Como análisis equivalente se implementan:
- **Curvas de aprendizaje:** muestran si el modelo tiene alto sesgo (underfitting) o alta varianza (overfitting).
- **Gráfico de residuos:** muestra la distribución del error de predicción.
- **MAE por categoría:** identifica subgrupos donde el modelo predice peor (análisis de errores diferencial).

---

## 4. Resultados Obtenidos

### Métricas en conjunto de prueba (hold-out 20%)

| Modelo | MAE | MSE | R² |
|--------|-----|-----|----|
| RegresionLineal | ~5,787 | ~64,243,252 | ~0.007 |
| **BosqueAleatorio** | ~5,838 | ~51,657,495 | **~0.201** |

**Mejor modelo: BosqueAleatorio** (seleccionado por R²).

### Validación cruzada 5-fold

| Modelo | R² Promedio | R² Std | MAE Promedio | MAE Std |
|--------|-------------|--------|--------------|---------|
| RegresionLineal | ~0.007 | ~0.02 | ~5,800 | ~300 |
| **BosqueAleatorio** | **~0.18** | **~0.04** | **~5,900** | **~400** |

El R² de BosqueAleatorio es consistentemente superior en todos los folds. La desviación estándar indica estabilidad razonable para un dataset de retail con alta varianza natural.

### Interpretación del R² = 0.20

Un R² de 0.20 significa que el modelo explica el 20% de la varianza de la demanda semanal. El 80% restante corresponde a variabilidad aleatoria inherente al comportamiento de compra de los clientes que no puede capturarse con solo 3 features temporales. Para el objetivo del sistema (identificar las categorías de **mayor** demanda relativa y priorizar el reabastecimiento), este nivel de predicción es suficiente — no se requiere predecir el volumen exacto, sino el ranking entre categorías.

---

## 5. Análisis de Errores por Categoría

El MAE varía entre categorías. Categorías como `Movies`, `Outdoor` y `Home & Kitchen` tienen un MAE hasta 11% superior al promedio global. Esto se analiza en profundidad en `docs/Modulos/ethics_analysis.md` como parte del análisis de fairness del Módulo E.

---

## 6. Comparación de Modelos — Trade-offs

| Criterio | RegresionLineal | BosqueAleatorio |
|----------|----------------|-----------------|
| R² | ~0.007 | **~0.20** |
| MAE | **~5,787** | ~5,838 |
| Tiempo de entrenamiento | < 1 s | ~2–5 s |
| Interpretabilidad | Alta (coeficientes) | Media (feature importance) |
| Manejo de no-linealidad | No | **Sí** |
| Sensible a escala de features | Sí | No |
| Robusto a outliers | No | **Sí** |

**Decisión final:** BosqueAleatorio es el modelo principal porque su R² es 28× mayor que RegresionLineal, demostrando que la demanda semanal tiene componentes no lineales que la regresión lineal no puede capturar. El costo de ~5 segundos de entrenamiento adicional es irrelevante en un sistema batch.

La Regresión Lineal se mantiene como **baseline obligatorio** para cuantificar la ganancia del modelo más complejo.

---

## 7. Conexión con Módulos Adyacentes

### Módulo B → Módulo E (y de ahí a Módulo A)

Módulo B entrega al pipeline de Módulo E:
1. El modelo entrenado (BosqueAleatorio).
2. Un diccionario de predicciones por categoría para la última semana.

```python
# Esto lo hace Módulo E en pipeline.py — no es responsabilidad de B
predicciones = {
    'Toys':   53311.26,
    'Tools':  53273.84,
    'Sports': 53236.42,
    ...
}
top3 = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:3]
# → pasa top3 al Módulo A
```

La orquestación que conecta las predicciones de B con el Módulo A es responsabilidad de **Módulo E**.

### Módulo B → Módulo C

```python
# En train.py — EntrenadorDemanda.generar_modelos_entrenados(incluir_dl=True)
X_entrenamiento  # shape: (710, 3) — mismo que usa RegresionLineal y BosqueAleatorio
y_entrenamiento  # shape: (710,)   — DemandaTotal semanal
→ RedNeuronalDensa.entrenar(X_entrenamiento, y_entrenamiento)
→ RedLSTM.preparar_datos(y_entrenamiento)  # serie temporal para LSTM
```

El Módulo B es la fuente de datos única para todo el pipeline predictivo. Módulo C no accede al CSV directamente; recibe los datos ya limpios, agregados y codificados por Módulo B.

---

## 8. Limitaciones Conocidas y Trabajo Futuro

| Limitación | Impacto | Mitigación Posible |
|------------|---------|-------------------|
| Solo 3 features | R² limitado (~0.20) | Agregar lag features (demanda semana anterior) |
| LabelEncoder en modelo lineal | Ordinaidad falsa en categorías | OneHotEncoder para regresión lineal |
| 9 meses de datos (Jan–Sep 2024) | No captura ciclos anuales completos | Reentrenar con datos de año completo |
| Agrupación por semana pierde patrones intra-semana | No predice demanda por día | Modelo adicional de granularidad diaria |
| Sin normalización de features | Coeficientes lineales no comparables entre features | Agregar StandardScaler como paso previo a RegresionLineal |

---

## 9. Reproducibilidad

```bash
# Ejecutar solo el pipeline ML desde la raíz del proyecto
python -c "
import sys
sys.path.insert(0, 'src')
from data.load_data import RepositorioVentas
from ml.preprocess import ProcesadorDemanda
from ml.train import EntrenadorDemanda
from ml.evaluate import EvaluadorModelos

datos = RepositorioVentas('Retail_Sales_Data.csv').cargar_datos_estructurados()
proc = ProcesadorDemanda(datos)
dp = proc.ejecutar_transformacion()
print(f'Nulos removidos: {proc.nulos_removidos}')
print(f'Muestras procesadas: {len(dp)}')

entrenador = EntrenadorDemanda(dp, ['Mes','SemanaDelAno','Categoria_Codificada'], 'DemandaTotal')
modelos, X_test, y_test = entrenador.generar_modelos_entrenados(incluir_dl=False)
metricas = EvaluadorModelos(modelos, X_test, y_test).calcular_metricas()

cv = entrenador.validar_cruzado(k=5)

print()
print('Hold-out 20%:')
for nombre, m in metricas.items():
    print(f'  {nombre}: MAE={m[\"MAE\"]:.2f}  R²={m[\"R2\"]:.4f}')
print()
print('Validacion cruzada 5-fold:')
for nombre, c in cv.items():
    print(f'  {nombre}: R²={c[\"R2_promedio\"]} ± {c[\"R2_std\"]}')
"

# Ejecutar demo interactivo (notebook)
jupyter notebook src/demo/ml_analysis.ipynb
```

---

*Documentación generada para defensa oral — Semana Final, 26–30 mayo 2026.*  
*Responsable: José Avila — Módulo B, Pipeline ML.*
