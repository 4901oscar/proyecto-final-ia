def generate_sales_summary(data): #entrada del deep learning de app.py genera el texto automatico

    ventas = data.get("ventas_totales") #extrae los datos que se le pasan desde deep los totales
    categoria = data.get("categoria_top") #extrae la categoria con mas ventas
    crecimiento = data.get("crecimiento") #extrae el crecimiento porcentual de las ventas

    if crecimiento > 0:
        trend = "crecimiento"
    elif crecimiento < 0:
        trend = "disminución"
    else:
        trend = "estabilidad"

# Se genera un resumen basado en los datos proporcionados basado en el deep learning
    summary = (
        f"La categoría {categoria} registró ventas estimadas de "
        f"{ventas} unidades con una tendencia de {trend} "
        f"del {abs(crecimiento)}%."
    )

    return {
        "summary": summary
    }
