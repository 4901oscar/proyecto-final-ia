---
marp: true
theme: default
paginate: true
header: "Flujo de Ejecución y Resultados Analíticos"
footer: "Proyecto Final IA - Mayo 2026"
backgroundColor: #ffffff
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #2980b9;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 5px;
  }
  code {
    background: #f4f6f7;
    color: #c0392b;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .box {
    background: #ecf0f1;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 10px;
  }
---

# Flujo de Ejecución y Resultados del Sistema
## Cómo funciona el Pipeline Integrado (End-to-End)

**Proyecto Final — Inteligencia Artificial**
2026

---

## 1. El Flujo de Ejecución (Pipeline)

El sistema se ejecuta mediante un orquestador central (`pipeline.py`) que encadena los módulos en el siguiente orden:

<div style="text-align: center; font-size: 0.9em; margin-top: 20px;">
  <div class="box">1. Ingesta de Datos (100k registros)</div>
  ⬇️
  <div class="box">2. Preprocesamiento & Feature Engineering (ML)</div>
  ⬇️
  <div class="box">3. Entrenamiento Híbrido (Random Forest + Redes Neuronales)</div>
  ⬇️
  <div class="box">4. Búsqueda A* (Rutas basadas en demanda) + Análisis NLP</div>
  ⬇️
  <div class="box">5. Evaluación de Equidad (Fairness) & Reporte</div>
</div>

---

## 2. Ingesta y Preprocesamiento (Fase 1)

**¿Cómo funciona?**
- Se carga el archivo `Retail_Sales_Data.csv`.
- El `ProcesadorDemanda` agrupa transacciones diarias.
- **Transformaciones:** 
  - Codificación de categorías (One-Hot / Label).
  - Extracción temporal (Mes, Día de la Semana).
- **Salida:** Un dataset estructurado donde el "Target" es la `DemandaTotal` histórica por categoría.

---

## 3. Predicción de Demanda (Fases 2 y 3)

El sistema no confía en un solo modelo. Utiliza un enfoque de **Ensemble**.

<div class="columns">
<div>

**Ejecución Paralela:**
1. **ML Clásico:** Random Forest entrena sobre features estructurados.
2. **Deep Learning:** 
   - *MLP (Densa)* captura patrones no lineales.
   - *LSTM* analiza la ventana temporal (últimos 7 días).

</div>
<div>

**Resultados generados:**
- Array de predicciones por categoría.
- Métricas cruzadas: MAE global de ~2,800 unidades.
- Las prioridades de inventario se actualizan dinámicamente.

</div>
</div>

---

## 4. Agente de Rutas en Acción (Fase 4)

Una vez que ML/DL predicen qué productos se agotarán, el módulo de búsqueda entra en acción.

**El Proceso:**
1. Se mapea el almacén en un grid lógico (8x8).
2. Los estantes de categorías con alta demanda predicha se marcan como "Objetivos de Alta Prioridad".
3. El agente evalúa nodos usando $f(n) = g(n) + DistanciaManhattan(n, objetivo)$.

**Resultado:** Una lista ordenada de coordenadas (ej. `[(0,0), (0,1), (1,1)...]`) que minimiza los pasos del operario.

---

## 5. Análisis Lingüístico y Resúmenes (NLP)

Mientras el agente calcula rutas, el módulo NLP interpreta el estado del negocio.

**Generación de Resultados:**
- Recibe las predicciones de ML/DL.
- Genera resúmenes ejecutivos automáticos (NLG).
- Evalúa el sentimiento del texto mediante un LLM (`BERT Multilingüe`).

**Ejemplo de Output generado:**
> *"La categoría Beauty registró ventas estimadas de 15,000 unidades con una tendencia de crecimiento del 12%. El sentimiento del cliente es clasificado como POSITIVO (4.5/5 estrellas)."*

---

## 6. Resultados del Análisis: Equidad y Sesgo (Fase 5)

El análisis final (`ethics_analysis`) genera un reporte crítico sobre el comportamiento del algoritmo:

- **Segmentación:** Divide el MAE de predicción cruzándolo con datos demográficos (Género).
- **Hallazgo Empírico generado:** 
  - Categoría "Tools" (Dominancia Masculina): MAE +21.9% sobre la media.
  - Categoría "Beauty" (Dominancia Femenina): MAE +13.8% sobre la media.
- **Acción:** El reporte imprime alertas para advertir que el desabastecimiento afectará desproporcionadamente a ciertos subgrupos.

---

## 7. Ejecución Práctica

Todo este flujo complejo se abstrae para el usuario final.

**Comando de ejecución:**
```bash
python src/integration/pipeline.py
```

**Salida en consola:**
1. ✅ `[OK] Datos cargados (100,000 filas).`
2. 🧠 `[TRAIN] Entrenando Random Forest... R2: 0.85.`
3. 🤖 `[DL] Entrenando LSTM (Ventana=7)...`
4. 🗺️ `[A*] Ruta óptima calculada: 14 pasos.`
5. 📊 `[NLP] Resumen generado.`
6. ⚖️ `[ÉTICA] Advertencia de Sesgo: Categoría 'Tools'.`

---

# ¡Gracias!
## Sistema listo para demostración en vivo.
