failure_cases = [
    {
        "text": "Este producto está mortal",
        "problem": "El modelo puede interpretar la frase como negativa aunque significa algo positivo."
    },
    {
        "text": "Excelente servicio... nunca llegó el pedido",
        "problem": "El sarcasmo puede generar errores de clasificación."
    }
]

def get_failure_cases():
    return failure_cases
