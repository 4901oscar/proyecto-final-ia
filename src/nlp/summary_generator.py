def generate_sales_summary(data):

    ventas = data.get("ventas_totales")
    categoria = data.get("categoria_top")
    crecimiento = data.get("crecimiento")

    if crecimiento > 0:
        trend = "crecimiento"
    elif crecimiento < 0:
        trend = "disminución"
    else:
        trend = "estabilidad"

    summary = (
        f"La categoría {categoria} registró ventas estimadas de "
        f"{ventas} unidades con una tendencia de {trend} "
        f"del {abs(crecimiento)}%."
    )

    return {
        "summary": summary
    }
