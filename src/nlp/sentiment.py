from transformers import pipeline


#se usa el modelo llm


#recibe el texto original del app.py
#devuelve un jason del texto enviado
classifier = pipeline( #nlp
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment" #llm
)

def analyze_sentiment(text):
    result = classifier(text) #analizael texto y devuelve el resultado del modelo llm (sentiment)

    return {
        "label": result[0]["label"],
        "score": float(result[0]["score"])
    }
