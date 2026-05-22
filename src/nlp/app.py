from fastapi import FastAPI #importacion de FastAPI para crear la aplicación web y definir las rutas de la API
from pydantic import BaseModel #importacion para definir modelos de datos que se utilizaran en las solicitudes y respuestas de la API

from preprocess import preprocess_text
from sentiment import analyze_sentiment
from summary_generator import generate_sales_summary
from evaluator import get_failure_cases

app = FastAPI() #define la creacion del servidor y las rutas

#se revice el JSON manual, la entrada es guardada en review.text
class ReviewRequest(BaseModel):
    texto: str

#entreada desde el deep learning
class SalesPredictionRequest(BaseModel):
    ventas_totales: int
    categoria_top: str
    crecimiento: float

@app.get("/")
def home():
    return {"message": "Microservicio NLP/LLM funcionando"}

#entrada manual del swagger
@app.post("/analyze")
def analyze_review(review: ReviewRequest): #recibe el texto manual lo procesa y devuelve los tokens

    tokens = preprocess_text(review.texto) #procesa el texto de entrada manual

    sentiment = analyze_sentiment(review.texto) #manda el texto al modelo nlp

#se devuelve al usuario el resultado del nlp
    return {
        "texto_original": review.texto,
        "tokens": tokens,
        "sentimiento": sentiment
    }


@app.post("/generate-summary")
def generate_summary(data: SalesPredictionRequest): #recibe los datos de deep learning

    summary = generate_sales_summary(data.dict()) #uso de los datos que proporciona deep learning lo manda a summary_genrator.py

    return summary

@app.get("/evaluation")
def evaluation_cases():
    return get_failure_cases()
