# Proyecto de ia, primeras implementaciones de NLP / LLM Integrado

## Funcionalidades
- NLP
- Sentiment Analysis
- Generación automática de resúmenes
- Integración con Deep Learning

## Endpoints
- POST /analyze
- POST /generate-summary
- GET /evaluation


## para probarlo manual se puede seguir este flujo


 -se puede usar este JSON para poder probarlo
-{
-"ventas_totales": 250000,
-"categoria_top": "Electronics",
-"crecimiento": -5
-}

- http://127.0.0.1:8000/docs
- swagger
- post/analyze
- app.py
- preprocessing.py
- sentiment.py
- requests JSON


## para probarlo con deep learning se puede seguir este flujo

-http://127.0.0.1:8000
-post/generate-summary
-app.py
-summary_generation.py
-requests JSON

## Entrada que se espera de deep learning
-data = {
-"ventas_totales": 250000,
-"categoria_top": "Electronics",
-"crecimiento": -5
-}