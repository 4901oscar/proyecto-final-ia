from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def analyze_sentiment(text):
    result = classifier(text)

    return {
        "label": result[0]["label"],
        "score": float(result[0]["score"])
    }
