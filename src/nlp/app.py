from fastapi import FastAPI #importacion para crear la api rest
from pydantic import BaseModel

from preprocess import preprocess_text
from sentiment import analyze_sentiment
from summary_generator import generate_sales_summary
from evaluator import get_failure_cases

app = FastAPI()

class ReviewRequest(BaseModel):
    texto: str

class SalesPredictionRequest(BaseModel):
    ventas_totales: int
    categoria_top: str
    crecimiento: float

@app.get("/")
def home():
    return {"message": "Microservicio NLP/LLM funcionando"}

@app.post("/analyze")
def analyze_review(review: ReviewRequest):

    tokens = preprocess_text(review.texto)

    sentiment = analyze_sentiment(review.texto)

    return {
        "texto_original": review.texto,
        "tokens": tokens,
        "sentimiento": sentiment
    }

@app.post("/generate-summary")
def generate_summary(data: SalesPredictionRequest):

    summary = generate_sales_summary(data.dict())

    return summary

@app.get("/evaluation")
def evaluation_cases():
    return get_failure_cases()
