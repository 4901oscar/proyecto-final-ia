import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))

#proces el tecto de entrada manual y lo transforma en token
#lo manda el endpont de analyze desde app.py
def preprocess_text(text):
    text = text.lower() #convierte el texto a minúsculas para normalizarlo
    text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', text) #elimina simbolos
    tokens = word_tokenize(text) # aca se dividen los textos en palabras individuales (tokens)
    filtered_tokens = [word for word in tokens if word not in stop_words] #elimina las palabras vacías
    return filtered_tokens #devuelve los tokens con las palabras que se desean
