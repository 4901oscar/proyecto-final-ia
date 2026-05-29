---
marp: true
theme: default
paginate: true
header: "Sistema de Optimización Inteligente de Inventario y Rutas"
footer: "Proyecto Final IA - Mayo 2026"
backgroundColor: #f8f9fa
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #34495e;
    border-bottom: 2px solid #3498db;
  }
  code {
    background: #eef1f4;
    color: #c0392b;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

# Sistema de Optimización Inteligente de Inventario y Rutas de Despacho
## Integración de Búsqueda Heurística, ML, DL y NLP

**Proyecto Final — Inteligencia Artificial**
2026

---

## 1. El Desafío de Negocio

El sistema aborda la ineficiencia logística en retail mediante:

- **Predicción de Demanda:** ¿Qué categorías se agotarán pronto?
- **Optimización de Rutas:** ¿Cómo minimizar el tiempo de recolección en almacén?
- **Análisis de Cliente:** ¿Cuál es la percepción cualitativa de los productos?
- **Equidad (Fairness):** ¿El sistema penaliza a ciertos segmentos demográficos?

---

## 2. Arquitectura del Sistema (Modular)

El sistema se divide en 5 módulos interconectados:

1.  **Módulo A (Search/CSP):** Algoritmo A* para rutas.
2.  **Módulo B (ML):** Modelos de regresión de demanda.
3.  **Módulo C (Deep Learning):** Redes MLP y LSTM.
4.  **Módulo D (NLP):** Sentimiento con BERT multilingüe.
5.  **Módulo E (Integración/Ética):** Pipeline y Fairness.

---

## 3. Módulo A: Optimización con A*

<div class="columns">
<div>

- **Algoritmo:** Búsqueda A*.
- **Costo:** $f(n) = g(n) + h(n)$.
- **Heurística:** Distancia Manhattan (admisible).
- **Entorno:** Grid de 8x8 que representa el layout del almacén.

</div>
<div>

```python
def manhattan_distance(state, goal):
    return abs(state[0] - goal[0]) + \
           abs(state[1] - goal[1])

# Prioriza estantes según 
# la demanda predicha
```

</div>
</div>

---

## 4. Módulo B: Machine Learning

- **Pipeline:** Preprocesamiento $\rightarrow$ Entrenamiento $\rightarrow$ Evaluación.
- **Modelos:** Random Forest Regressor y Regresión Lineal.
- **Validación:** K-Fold Cross-Validation ($k=5$).
- **Features clave:** 
    - Estacionalidad (Mes, Día de la semana).
    - Codificación de categorías.
- **Meta:** Predecir `DemandaTotal` diaria por categoría.

---

## 5. Módulo C: Deep Learning

<div class="columns">
<div>

### Red Densa (MLP)
- Arquitectura: `[128, 64, 32]`.
- Activación: ReLU.
- Optimizador: Adam con Early Stopping.

</div>
<div>

### Red LSTM (Temporal)
- Ventana temporal de 7 días.
- Unidades LSTM: 64.
- Captura dependencias secuenciales en las ventas.

</div>
</div>

---

## 6. Módulo D: NLP (BERT Multilingüe)

Análisis de la percepción del cliente para ajustar prioridades de inventario.

- **Modelo:** `nlptown/bert-base-multilingual-uncased-sentiment`.
- **Salida:** Clasificación de sentimiento (1-5 estrellas).
- **Generación de Resúmenes:** Motor lógico que traduce predicciones numéricas a reportes ejecutivos legibles.

---

## 7. Módulo E: Análisis de Ética y Sesgo

### Hallazgos de Fairness
- **Sesgo Detectado:** Categorías como *Tools* y *Beauty* tienen un **MAE un 20% superior** al promedio global.
- **Correlación:** Existe una relación directa entre el sesgo de representación demográfica (género) y el error del modelo.
- **Mitigación:** Implementación de técnicas de imputación y balanceo para reducir el "Omitted Variable Bias".

---

## 8. Conclusiones

1.  **Integración Exitosa:** Los 5 módulos operan en un pipeline único sincronizado.
2.  **Rutas Inteligentes:** El operario reduce distancias recorridas gracias a la heurística informada por la demanda de ML.
3.  **Transparencia:** El análisis de ética garantiza un sistema auditable y justo.
4.  **Escalabilidad:** Arquitectura preparada para despliegue en microservicios.

---

# ¡Gracias!
## ¿Preguntas?
