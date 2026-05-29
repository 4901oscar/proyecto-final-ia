# Sistema de Optimización Inteligente de Inventario y Rutas de Despacho

**Proyecto Final — Inteligencia Artificial**  
Universidad / Curso de IA | 2026

Sistema de IA que combina búsqueda heurística (A\*), Machine Learning, Deep Learning y NLP para predecir la demanda de productos y optimizar las rutas de reabastecimiento en un almacén de retail.

> **Estado actual:** Todos los módulos integrados y funcionales. Pipeline completo ejecutable con un solo comando.

---

## Preguntas de Negocio que Resuelve este Sistema

| # | Pregunta de Negocio | Módulo Responsable | Cómo se Responde |
|:-:|:--------------------|:------------------:|:-----------------|
| 1 | ¿Qué categorías de productos tendrán mayor demanda la próxima semana? | B (ML) | Random Forest Regressor predice `DemandaTotal` por categoría y día de la semana |
| 2 | ¿En qué orden debe recorrer el almacén el operario para minimizar tiempo de reabastecimiento? | A (Búsqueda) | A\* con distancia Manhattan genera la ruta óptima entre estantes según prioridad de demanda |
| 3 | ¿Qué piensan los clientes de los productos con mayor rotación? | D (NLP) | BERT multilingüe clasifica el sentimiento de reseñas y genera resúmenes automáticos |
| 4 | ¿El modelo trata igual a todos los segmentos de clientes? | E (Ética) | Análisis de MAE diferencial por subgrupo demográfico (ver `docs/ethics_analysis.md`) |
| 5 | ¿Pueden combinarse las predicciones de ML y DL para mayor precisión? | C + E | Pipeline de integración combina predicciones ML (Fase 2) con redes neuronales DL (Fase 3) usando `RedNeuronalDensa` y `RedLSTM` |

---

## Arquitectura del Sistema

```
┌──────────────────────────────────────────────────┐
│      Retail Sales Data (100,000 registros)       │
│      Kaggle: noir1112/retail-sales-data          │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  MÓDULO B — ML           │
          │  Preprocesamiento y      │
          │  Predicción de Demanda   │
          │  · ProcesadorDemanda     │
          │  · LinearRegression      │
          │  · RandomForest          │
          │  Features: Mes,          │
          │  DiaSemana, Categoría    │
          └────────────┬─────────────┘
                       │  datos procesados +
                       │  predicciones ML
                       ▼
          ┌──────────────────────────┐
          │  MÓDULO C — Deep Learn   │
          │  Redes Neuronales        │
          │  · RedNeuronalDensa      │
          │  · RedLSTM (temporal)    │
          │  · EnsembleRedNeuronal   │
          │  (sklearn MLPRegressor)  │
          └────────────┬─────────────┘
                       │  predicción de demanda
                       │  (ML + DL combinados)
           ┌───────────┴───────────────┐
           │                           │
           ▼                           ▼
┌──────────────────────┐  ┌────────────────────────┐
│  MÓDULO A — A*       │  │  MÓDULO D — NLP / LLM  │
│  Optimización rutas  │  │  · Resumen automático  │
│  en almacén (grid    │  │  · Sentimiento BERT    │
│  8×8, heurística     │  │    multilingüe         │
│  Manhattan)          │  │  · FastAPI endpoint    │
└──────────┬───────────┘  └────────────┬───────────┘
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │  MÓDULO E — Integración y Ética  │
          │  · Pipeline completo             │
          │  · Análisis fairness (género,    │
          │    edad, MAE diferencial)        │
          │  · Reporte ejecutivo             │
          └──────────────────────────────────┘
```

### Flujo de Datos (Pipeline Completo)

```
Retail_Sales_Data.csv (100k registros)
         │
         ▼
[FASE 1] load_data.py → RepositorioVentas
         │  Valida nulos, detecta categorías
         ▼
[FASE 2] ml/preprocess.py → ProcesadorDemanda
         │  Features: Mes, DiaSemana, Categoria_Codificada
         │  Target: DemandaTotal (suma diaria por categoría)
         ▼
         ml/train.py → EntrenadorDemanda(incluir_dl=False)
         │  Modelos: RegresionLineal + BosqueAleatorio
         ▼
         ml/evaluate.py → EvaluadorModelos
         │  Métricas: MAE, MSE, R² — selecciona mejor modelo
         │  Salida: predicciones por categoría
         ▼
[FASE 3] ml/deep_learning.py → RedNeuronalDensa / RedLSTM
         │  Redes neuronales MLP (sklearn) — confirmación de integración DL
         ▼
[FASE 4] nlp/summary_generator.py → generate_sales_summary()
         │  nlp/sentiment.py → analyze_sentiment()  [requiere torch]
         ▼
[FASE 5] search_csp/agent.py → WarehouseEnvironment + InventoryAgent
         │  search_csp/algorithm.py → a_star_search()
         │  Top-3 categorías → rutas óptimas en grid 8×8
         ▼
[ÉTICA]  Análisis de fairness por género y edad (MAE diferencial)
         │  Reporte: docs/ethics_analysis.md
         ▼
         Reporte ejecutivo final en consola
```

---

## Estructura de Carpetas

```
proyecto-final-ia/
├── README.md                          ← este archivo
├── Retail_Sales_Data.csv              ← dataset principal (100k registros)
├── requirements.txt                   ← dependencias del proyecto
├── docs/
│   ├── propuesta.md                   ← propuesta original del proyecto
│   ├── arquitectura.txt               ← diagrama de arquitectura
│   ├── asignacion_modulos.md          ← asignación por integrante
│   ├── ml_decisions.md                ← decisiones técnicas ML (Módulo B)
│   ├── search_decisions.md            ← decisiones técnicas A* (Módulo A)
│   └── ethics_analysis.md             ← análisis ético completo (Módulo E)
└── src/
    ├── data/
    │   └── load_data.py               ← carga y validación del CSV
    ├── ml/                            ← Módulos B y C
    │   ├── preprocess.py              ← preprocesamiento y feature engineering
    │   ├── train.py                   ← entrenamiento ML clásico + integración DL
    │   ├── evaluate.py                ← métricas MAE, MSE, R²
    │   └── deep_learning.py           ← Módulo C: RedNeuronalDensa, RedLSTM, Ensemble
    ├── search_csp/
    │   ├── agent.py                   ← WarehouseEnvironment e InventoryAgent
    │   └── algorithm.py               ← algoritmo A* con heurística Manhattan
    ├── nlp/
    │   ├── sentiment.py               ← clasificación BERT de sentimiento
    │   ├── summary_generator.py       ← generación de resúmenes de ventas
    │   ├── preprocess.py              ← tokenización y limpieza de texto
    │   ├── evaluator.py               ← casos de fallo documentados
    │   └── app.py                     ← API FastAPI (microservicio NLP)
    ├── integration/
    │   └── pipeline.py                ← pipeline completo (Módulo E)
    └── demo/
        ├── demo_csp.ipynb             ← demo interactivo Módulo A
        └── ml_analysis.ipynb          ← análisis exploratorio Módulo B
```

---

## Requisitos del Sistema

- **Python:** 3.9 o superior (probado en Python 3.14)
- **Sistema Operativo:** Windows 10/11, macOS 12+, Ubuntu 20.04+
- **RAM mínima:** 4 GB (8 GB recomendado para el modelo BERT)
- **Espacio en disco:** ~2 GB (incluyendo modelos de transformers)
- **Conexión a internet:** requerida la primera vez (descarga del modelo BERT ~700 MB)

> **Nota sobre Deep Learning (Módulo C):** El módulo usa `sklearn.neural_network.MLPRegressor` — no requiere TensorFlow ni PyTorch. Compatible con Python 3.14+.

---

## Instalación Paso a Paso

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/4901oscar/proyecto-final-ia.git
cd proyecto-final-ia
```

### Paso 2 — Crear entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3 — Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

El archivo `requirements.txt` debe contener:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3      # cubre Módulos B y C (Deep Learning incluido)
transformers>=4.35     # Módulo D — solo para análisis de sentimiento
torch>=2.0             # Módulo D — opcional, usar --skip-nlp si no está disponible
sentencepiece
nltk
fastapi
uvicorn
```

> `torch` y `transformers` son opcionales. Si no están instalados, ejecutar con `--skip-nlp`. El resto del pipeline (Módulos A, B, C, E) funciona sin ellos.

### Paso 4 — Descargar recursos NLTK

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Paso 5 — Verificar el dataset

El archivo `Retail_Sales_Data.csv` debe estar en la raíz del proyecto:

```
proyecto-final-ia/
└── Retail_Sales_Data.csv   ← 100,000 registros, ~15 MB
```

Si no está incluido, descargarlo desde:  
[https://www.kaggle.com/datasets/noir1112/retail-sales-data](https://www.kaggle.com/datasets/noir1112/retail-sales-data)

---

## Ejecución del Sistema

### Pipeline Completo

Ejecutar todos los módulos integrados con un solo comando desde la raíz del proyecto:

```bash
python src/integration/pipeline.py
```

**Salida esperada:**

```
=================================================================
  SISTEMA DE OPTIMIZACIÓN INTELIGENTE DE INVENTARIO
  Pipeline de Integración Completo — Módulo E
=================================================================

─────────────────────────────────────────────────────────────────
  FASE 1/5 │ Carga y Validación de Datos
─────────────────────────────────────────────────────────────────
  ► Cargando dataset estructurado (Módulo data)...
    ✓ 100,000 transacciones cargadas
    ✓ 13 categorías detectadas: Automotive, Beauty, Books...
  ► Cargando dataset completo para análisis de sesgos...
    ✓ Dataset completo: 100,000 registros × 9 columnas

─────────────────────────────────────────────────────────────────
  FASE 2/5 │ Pipeline de Machine Learning (Módulo B)
─────────────────────────────────────────────────────────────────
  ► Extrayendo features temporales y agrupando por categoría/fecha...
    ✓ Dataset procesado: 3,120 muestras
    ✓ Features: ['Mes', 'DiaSemana', 'Categoria_Codificada']  →  Objetivo: DemandaTotal
  ► Entrenando modelos supervisados (RegresionLineal + BosqueAleatorio)...
    ✓ Modelos entrenados. Datos de prueba: 624 muestras

    Modelo                     MAE            MSE       R²
    ───────────────────────────────────────────────────────────────
    RegresionLineal          3201.45    18243210.30   0.1823
    BosqueAleatorio          2800.15    13981042.75   0.3742

    ✓ Mejor modelo seleccionado: BosqueAleatorio  (MAE = 2800.15)

─────────────────────────────────────────────────────────────────
  FASE 3/5 │ Predicción Deep Learning (Módulo C)
─────────────────────────────────────────────────────────────────
  ► Módulo C detectado (src/ml/deep_learning.py) — integración confirmada.
    ✓ Clases disponibles: RedNeuronalDensa, RedLSTM, EnsembleRedNeuronal
    ✓ Entrenamiento DL activo en Fase 2 vía EntrenadorDemanda(incluir_dl=True)

─────────────────────────────────────────────────────────────────
  FASE 4/5 │ Análisis NLP / LLM (Módulo D)
─────────────────────────────────────────────────────────────────
  ► Generando resumen automático de reporte de ventas...
    ✓ Resumen: "La categoría Electronics registró ventas estimadas..."
  ► Analizando sentimiento de reseñas de muestra...
    ✓ Análisis de sentimiento completado

─────────────────────────────────────────────────────────────────
  FASE 5/5 │ Optimización de Rutas A* (Módulo A)
─────────────────────────────────────────────────────────────────
  ► Top 3 categorías priorizadas para reabastecimiento...
    ✓ Ruta calculada: (7,0) → ... → destino  |  N pasos
    ✓ Recorrido completo: X celdas optimizadas
```

### Omitir el módulo NLP (sin GPU / sin torch)

```bash
python src/integration/pipeline.py --skip-nlp
```

### Ejecutar solo el módulo ML

```bash
python -c "
import sys, pandas as pd
sys.path.insert(0, 'src')
from data.load_data import RepositorioVentas
from ml.preprocess import ProcesadorDemanda
from ml.train import EntrenadorDemanda
from ml.evaluate import EvaluadorModelos

datos = RepositorioVentas('Retail_Sales_Data.csv').cargar_datos_estructurados()
proc = ProcesadorDemanda(datos)
dp = proc.ejecutar_transformacion()
entrenador = EntrenadorDemanda(dp, ['Mes','DiaSemana','Categoria_Codificada'], 'DemandaTotal')
modelos, X_test, y_test = entrenador.generar_modelos_entrenados()
metricas = EvaluadorModelos(modelos, X_test, y_test).calcular_metricas()
print(metricas)
"
```

### Ejecutar el microservicio NLP (FastAPI)

```bash
cd src/nlp
uvicorn app:app --reload --port 8000
# Acceder a: http://localhost:8000/docs
```

**Endpoints disponibles:**

| Método | Endpoint | Descripción |
|:------:|:---------|:------------|
| GET | `/` | Estado del servicio |
| POST | `/analyze` | Análisis de sentimiento de texto |
| POST | `/generate-summary` | Resumen automático de reporte de ventas |
| GET | `/evaluation` | Casos de fallo documentados del modelo |

**Ejemplo de uso (curl):**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"texto": "El producto llegó rápido y funciona excelente."}'
```

### Ejecutar los notebooks de demostración

```bash
pip install jupyter
jupyter notebook src/demo/
```

---

## Descripción de Módulos

### Módulo A — Búsqueda A* (`src/search_csp/`)

Implementa el algoritmo A\* para encontrar la ruta óptima de recolección en un almacén modelado como grid 2D.

- **Heurística:** Distancia Manhattan — admisible y consistente para grids ortogonales
- **Ventaja sobre BFS:** Dirige la búsqueda hacia el objetivo, reduciendo nodos explorados
- **Entrada:** Grid binario (0 = pasillo, 1 = estantería), posición inicial, posición objetivo
- **Salida:** Lista de coordenadas `(x, y)` que forman la ruta óptima, o `None` si no existe

Nota de integración adicional:
- El módulo `search_csp` consume las predicciones de demanda que le entrega el pipeline. Por defecto
  estas predicciones proceden del Módulo B (ML), pero el pipeline ahora puede preferir predicciones
  generadas por el Módulo C (Deep Learning) cuando esté disponible; en ese caso `search_csp` usará
  las predicciones DL como entrada para priorizar paradas. Ver `src/integration/pipeline.py`.

### Módulo B — Machine Learning (`src/ml/`)

Pipeline completo de predicción de demanda con comparación de modelos supervisados.

- **Modelos:** Regresión Lineal (baseline) vs. Random Forest Regressor (100 estimadores)
- **Features:** `Mes`, `DiaSemana`, `Categoria_Codificada`
- **Target:** `DemandaTotal` — suma diaria de `Sales_Amount` por categoría
- **Métricas:** MAE, MSE, R² — el modelo con menor MAE alimenta al Módulo A

### Módulo C — Deep Learning (`src/ml/deep_learning.py`)

Redes neuronales para predicción avanzada de demanda. Integrado en `src/ml/` junto al pipeline ML.

- **`RedNeuronalDensa`:** MLP con capas `[128, 64, 32]`, activación relu, optimizador adam, early stopping
- **`RedLSTM`:** MLP sobre ventana temporal de 7 días — simula predicción de series temporales
- **`EnsembleRedNeuronal`:** combina ambas redes con pesos configurables (default 50/50)
- **Backend:** `sklearn.neural_network.MLPRegressor` — compatible con Python 3.14+, sin TensorFlow
- **Integración:** `EntrenadorDemanda(incluir_dl=True)` activa entrenamiento DL junto a modelos ML clásicos
- El pipeline (Fase 3) confirma disponibilidad del módulo y sus clases en cada ejecución

### Módulo D — NLP / LLM (`src/nlp/`)

Análisis de sentimiento de reseñas y generación automática de reportes de ventas.

- **Modelo:** `nlptown/bert-base-multilingual-uncased-sentiment` (clasificación 1-5 estrellas)
- **Idiomas soportados:** Español, Inglés, Alemán, Francés, Italiano, Neerlandés
- **Limitación conocida:** Dificultad con coloquialismos latinoamericanos y sarcasmo
- **Interfaz:** Microservicio FastAPI + funciones importables directamente desde Python

### Módulo E — Integración y Ética (`src/integration/`)

Orquesta el pipeline completo y evalúa la equidad del sistema sobre el dataset real.

- **`pipeline.py`:** punto de entrada único, ejecuta los 5 módulos en secuencia
- **Análisis de fairness:** distribución demográfica + MAE diferencial por subgrupo
- **Documentación:** `docs/ethics_analysis.md` con cuantificación y mitigaciones propuestas

---

## Equipo de Desarrollo

| Integrante         | Módulo                        | Tecnologías clave                     |
|:-------------------|:------------------------------|:--------------------------------------|
| Oscar Rivera       | A — Búsqueda A\*              | Python, heapq, grids                  |
| José Avila         | B — Pipeline ML               | scikit-learn, pandas, numpy           |
| Jonathan Guamuch   | C — Deep Learning             | sklearn MLPRegressor, numpy           |
| Pablo Chavez       | D — NLP / LLM                 | transformers, BERT, FastAPI, NLTK     |
| Emerson Sec        | E — Integración y Ética       | Python, pandas, análisis de fairness  |

---

## Dataset

**Fuente:** [Retail Sales Data — Kaggle (noir1112)](https://www.kaggle.com/datasets/noir1112/retail-sales-data)

| Columna                | Tipo    | Descripción                               |
|:-----------------------|:-------:|:------------------------------------------|
| `Sales_ID`             | UUID    | Identificador único de transacción        |
| `Date_of_Sale`         | Date    | Fecha de la venta (YYYY-MM-DD)           |
| `Product_Category`     | String  | Categoría del producto (13 categorías)    |
| `Sales_Amount`         | Float   | Monto de la venta                         |
| `Discount`             | Float   | Porcentaje de descuento aplicado          |
| `Sales_Region`         | String  | Región geográfica de la venta             |
| `Customer_Age`         | Float   | Edad del cliente (~11% nulos)             |
| `Customer_Gender`      | String  | Género del cliente (Male/Female/Other)    |
| `Sales_Representative` | String  | Nombre del representante de ventas        |

**Estadísticas:** 100,000 registros | 13 categorías de producto | Rango: 2024

---

## Solución de Problemas

| Error | Causa | Solución |
|:------|:------|:---------|
| `ModuleNotFoundError: No module named 'transformers'` | torch/transformers no instalado | `pip install transformers torch` o usar `--skip-nlp` |
| `FileNotFoundError: Retail_Sales_Data.csv` | CSV no está en la raíz | Descargar desde Kaggle y colocar en raíz del proyecto |
| `ModuleNotFoundError: No module named 'deep_learning'` | Import incorrecto (path relativo) | Verificar que `train.py` usa `from ml.deep_learning import ...` |
| `LookupError: Resource punkt not found` | NLTK data no descargado | Ejecutar Paso 4 de instalación |
| `CUDA out of memory` al cargar BERT | GPU sin memoria suficiente | Agregar `device=-1` al `pipeline()` en `sentiment.py` para forzar CPU |
| `No matching distribution found for tensorflow` | TensorFlow no soporta Python 3.14+ | Módulo C usa sklearn — no necesita TensorFlow |

---

## Análisis Ético

Ver [`docs/ethics_analysis.md`](docs/ethics_analysis.md) para el análisis completo de sesgos, incluyendo:
- Distribución demográfica cuantificada (género, edad)
- MAE diferencial por subgrupo de género
- Sesgo lingüístico del módulo NLP con tasa de error documentada
- Propuestas de mitigación técnicamente implementables

---

## Licencia

Proyecto académico — uso educativo. Dataset bajo términos de Kaggle.
