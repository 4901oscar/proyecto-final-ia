# Documentación de Decisiones: Pipeline de Machine Learning (Módulo B)

## 1. Justificación de Modelos Elegidos

Para abordar el problema de predicción de demanda de inventario basado en el historial de ventas, el pipeline implementa y compara dos algoritmos supervisados de regresión:

*   **Regresión Lineal (Modelo Base):** Se seleccionó como línea base (*baseline*) debido a su alta explicabilidad y bajo costo computacional. Permite identificar si existe una relación lineal fuerte y directa entre las variables temporales (mes, día de la semana) y el volumen de ventas.
*   **Bosque Aleatorio (Random Forest Regressor):** Se eligió como modelo principal debido a la naturaleza del dominio de retail. La demanda de productos rara vez sigue un patrón estrictamente lineal; está sujeta a estacionalidades complejas y comportamientos de compra irregulares. Un ensamblado basado en árboles de decisión es altamente efectivo para capturar estas relaciones no lineales y es robusto frente a valores atípicos (*outliers*) presentes en transacciones comerciales.

## 2. Selección de Métricas y Análisis de Errores

Dado que la variable objetivo (`DemandaTotal` derivada de `Amount`) es continua, el problema se formula como una regresión multivariada. Las métricas de evaluación seleccionadas son:

*   **MAE (Error Absoluto Medio):** Proporciona la magnitud promedio de los errores en las predicciones en las mismas unidades de la demanda. Es altamente interpretable desde la perspectiva de negocio (ej. "nuestra predicción falla por un promedio de X unidades o quetzales").
*   **MSE (Error Cuadrático Medio):** Al elevar los errores al cuadrado, esta métrica penaliza severamente las predicciones que se alejan drásticamente del valor real. En logística, un error de predicción masivo (que resulte en un desabastecimiento total) es mucho más costoso que varios errores pequeños, por lo que minimizar el MSE es crítico.
*   **R² (Coeficiente de Determinación):** Indica qué porcentaje de la varianza en la demanda es explicada por el modelo (Mes, Día, Categoría).

*(Nota metodológica: Las métricas de precisión, recall, F1 y curvas ROC mencionadas en los lineamientos generales aplican a problemas de clasificación, por lo que fueron descartadas en favor de métricas de regresión adaptadas a este problema específico).*

## 3. Comparación Cuantitativa y Trade-offs

Al contrastar ambos modelos en la fase de evaluación, se establecen las siguientes conclusiones respecto a sus compensaciones (*trade-offs*):

*   **Complejidad vs. Precisión:** La Regresión Lineal ofrece inferencias casi instantáneas pero tiende a subajustar (*underfit*) los picos de demanda estacionales. Por otro lado, Random Forest minimiza significativamente el MAE y el MSE, capturando los picos de demanda de categorías específicas en días particulares de la semana, a cambio de un mayor costo de entrenamiento y memoria.
*   **Decisión Final:** Para la integración con el Agente de Búsqueda (Módulo A), se prioriza el modelo con el menor MAE (típicamente Random Forest), ya que una estimación más precisa de la cantidad de productos dictará de manera más eficiente la ruta y las paradas que debe realizar el agente en el almacén.