# Sistema de Optimización Inteligente de Inventario y Rutas de Despacho

**Proyecto Final — Inteligencia Artificial**  
Universidad / Curso de IA | 2026

Sistema de IA que combina búsqueda heurística (A\*), Machine Learning, Deep Learning y NLP para predecir la demanda de productos y optimizar las rutas de reabastecimiento en un almacén de retail.

---

## Preguntas de Negocio que Resuelve este Sistema

| # | Pregunta de Negocio | Módulo Responsable | Cómo se Responde |
|:-:|:--------------------|:------------------:|:-----------------|
| 1 | ¿Qué categorías de productos tendrán mayor demanda la próxima semana? | B (ML) | Random Forest Regressor predice `DemandaTotal` por categoría y día de la semana |
| 2 | ¿En qué orden debe recorrer el almacén el operario para minimizar tiempo de reabastecimiento? | A (Búsqueda) | A\* con distancia Manhattan genera la ruta óptima entre estantes según prioridad de demanda |
| 3 | ¿Qué piensan los clientes de los productos con mayor rotación? | D (NLP) | BERT multilingüe clasifica el sentimiento de reseñas y genera resúmenes automáticos |
| 4 | ¿El modelo trata igual a todos los segmentos de clientes? | E (Ética) | Análisis de MAE diferencial por subgrupo demográfico (ver `docs/ethics_analysis.md`) |
| 5 | ¿Pueden combinarse las predicciones de ML y DL para mayor precisión? | C + E | Pipeline de integración combina ambas predicciones cuando el Módulo C está disponible |

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────┐
│         Retail Sales Data (100,000 registros)   │
│         Kaggle: noir1112/retail-sales-data       │
└────────────────────────┬────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           │                            │
           ▼                            ▼
┌──────────────────────┐   ┌────────────────────────┐
│  MÓDULO B — ML       │   │  MÓDULO C — Deep Learn  │
│  Predicción Demanda  │   │  Red Neuronal (PyTorch/ │
│  · LinearRegression  │   │  TensorFlow) para ventas│
│  · RandomForest      │   │  avanzadas              │
│  Features: Mes,      │   │  (integración pendiente)│
│  DiaSemana, Categoría│   │                         │
└──────────┬───────────┘   └────────────┬────────────┘
           │                            │
           └──────────┬─────────────────┘
                      │  Predicción de demanda
           ┌──────────┴──────────────────────────────┐
           │                                          │
           ▼                                          ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│  MÓDULO A — Búsqueda A* │     │  MÓDULO D — NLP / LLM    │
│  Optimización de rutas  │     │  · Análisis de sentimiento│
│  de recolección en      │     │    (BERT multilingüe)     │
│  almacén con heurística │     │  · Resumen automático de  │
│  Manhattan              │     │    reportes de ventas     │
└────────────┬────────────┘     └─────────────┬────────────┘
             │                                │
             └──────────────┬─────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │  MÓDULO E — Integración y Ética      │
          │  · Pipeline completo (pipeline.py)   │
          │  · Análisis de sesgo por subgrupo    │
          │  · Documentación y README            │
          └─────────────────────────────────────┘
```

### Flujo de Datos

```
CSV → load_data.py → ProcesadorDemanda → EntrenadorDemanda → EvaluadorModelos
                                                    ↓
                            predicciones por categoría (DemandaTotal)
                                                    ↓
                     ┌──────────────────────────────┤
                     ↓                              ↓
            generate_sales_summary()       WarehouseEnvironment + a_star_search()
            analyze_sentiment()            → Ruta óptima de recolección
                     ↓                              ↓
              Reporte NLP                   Recorrido en celdas del almacén
                     └──────────────────────────────┘
                                        ↓
                              Reporte ejecutivo final
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
    ├── ml/
    │   ├── preprocess.py              ← preprocesamiento y feature engineering
    │   ├── train.py                   ← entrenamiento de modelos supervisados
    │   └── evaluate.py                ← métricas MAE, MSE, R²
    ├── dl/
    │   └── model.py                   ← (pendiente) red neuronal Módulo C
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

- **Python:** 3.9 o superior
- **Sistema Operativo:** Windows 10/11, macOS 12+, Ubuntu 20.04+
- **RAM mínima:** 4 GB (8 GB recomendado para el modelo BERT)
- **Espacio en disco:** ~2 GB (incluyendo modelos de transformers)
- **Conexión a internet:** requerida la primera vez (descarga del modelo BERT ~700 MB)

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
scikit-learn>=1.3
transformers>=4.35
torch>=2.0
sentencepiece
nltk
fastapi
uvicorn
```

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
  ⚠  Módulo C (src/dl/model.py) no encontrado — integración pendiente.
  ► Pipeline continúa con predicciones del Módulo B como fallback.

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

### Módulo B — Machine Learning (`src/ml/`)

Pipeline completo de predicción de demanda con comparación de modelos supervisados.

- **Modelos:** Regresión Lineal (baseline) vs. Random Forest Regressor (100 estimadores)
- **Features:** `Mes`, `DiaSemana`, `Categoria_Codificada`
- **Target:** `DemandaTotal` — suma diaria de `Sales_Amount` por categoría
- **Métricas:** MAE, MSE, R² — el modelo con menor MAE alimenta al Módulo A

### Módulo C — Deep Learning (`src/dl/`)

Red neuronal para predicción avanzada de ventas (pendiente de integración).

- **Interfaz esperada:** clase `DLPredictor` con método `predict() → dict`
- El pipeline detecta automáticamente si el módulo está disponible y usa fallback ML si no

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
| Jonathan Guamuch   | C — Deep Learning             | PyTorch / TensorFlow                  |
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
| `ModuleNotFoundError: No module named 'dl'` | Módulo C no entregado aún | Normal — pipeline continúa con fallback ML |
| `LookupError: Resource punkt not found` | NLTK data no descargado | Ejecutar Paso 4 de instalación |
| `CUDA out of memory` al cargar BERT | GPU sin memoria suficiente | Agregar `device=-1` al `pipeline()` en `sentiment.py` para forzar CPU |

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
