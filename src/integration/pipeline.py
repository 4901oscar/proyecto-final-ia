#!/usr/bin/env python3
"""
Pipeline de Integración Completo — Módulo E
Sistema de Optimización Inteligente de Inventario y Rutas de Despacho

Uso:
    python src/integration/pipeline.py
    python src/integration/pipeline.py --skip-nlp   # omite carga del modelo BERT
"""

import sys
import warnings
import argparse
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ─── Configuración de rutas del proyecto ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_PATH = PROJECT_ROOT / "Retail_Sales_Data.csv"

# ─── Mapa del almacén: 0 = pasillo libre, 1 = estantería (obstáculo) ─────────
WAREHOUSE_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

# Posición de estantes por categoría dentro del grid 8×8
CATEGORY_LOCATIONS = {
    "Electronics":      (1, 3),
    "Clothing":         (1, 6),
    "Groceries":        (3, 3),
    "Books":            (3, 6),
    "Beauty":           (5, 3),
    "Movies":           (5, 6),
    "Outdoor":          (1, 1),
    "Automotive":       (3, 1),
    "Tools":            (5, 1),
    "Toys":             (1, 7),
    "Pet Supplies":     (3, 7),
    "Office Supplies":  (5, 7),
    "DIY":              (6, 3),
    "Sports":           (6, 6),
}

WAREHOUSE_ENTRY = (7, 0)  # Muelle de carga / punto de inicio

FEATURES = ["Mes", "DiaSemana", "Categoria_Codificada"]
TARGET = "DemandaTotal"


# ─── Helpers de presentación ────────────────────────────────────────────────
def header(title: str, width: int = 65):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def step(msg: str):
    print(f"  ► {msg}")


def ok(msg: str):
    print(f"    ✓ {msg}")


def warn(msg: str):
    print(f"  ⚠  {msg}")


def fail(msg: str):
    print(f"  ✗  FALLO: {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Carga y Validación de Datos
# ═══════════════════════════════════════════════════════════════════════════════
def fase_carga_datos():
    header("FASE 1/5 │ Carga y Validación de Datos")

    if not DATA_PATH.exists():
        fail(f"Archivo CSV no encontrado en: {DATA_PATH}")
        sys.exit(1)

    try:
        from data.load_data import RepositorioVentas

        step("Cargando dataset estructurado (Módulo data)...")
        repo = RepositorioVentas(str(DATA_PATH))
        datos = repo.cargar_datos_estructurados()

        ok(f"{len(datos):,} transacciones cargadas")
        ok(
            f"{datos['Product_Category'].nunique()} categorías detectadas: "
            f"{', '.join(sorted(datos['Product_Category'].dropna().unique()[:5]))}..."
        )

        nulos = datos.isnull().sum()
        nulos_detectados = nulos[nulos > 0]
        if not nulos_detectados.empty:
            warn(f"Valores nulos: {nulos_detectados.to_dict()}")

        step("Cargando dataset completo para análisis de sesgos...")
        datos_completos = pd.read_csv(DATA_PATH)
        ok(f"Dataset completo: {len(datos_completos):,} registros × {datos_completos.shape[1]} columnas")

        return datos, datos_completos

    except ImportError as e:
        fail(f"No se pudo importar módulo de datos: {e}")
        sys.exit(1)
    except Exception as e:
        fail(f"Error inesperado al cargar datos: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Pipeline de Machine Learning (Módulo B)
# ═══════════════════════════════════════════════════════════════════════════════
def fase_ml(datos: pd.DataFrame):
    header("FASE 2/5 │ Pipeline de Machine Learning (Módulo B)")

    try:
        from ml.preprocess import ProcesadorDemanda
        from ml.train import EntrenadorDemanda
        from ml.evaluate import EvaluadorModelos
    except ImportError as e:
        fail(f"No se pudo importar Módulo B (ML): {e}")
        sys.exit(1)

    # Preprocesamiento
    step("Extrayendo features temporales y agrupando por categoría/fecha...")
    try:
        procesador = ProcesadorDemanda(datos)
        datos_proc = procesador.ejecutar_transformacion()
    except Exception as e:
        fail(f"Error en preprocesamiento ML: {e}")
        sys.exit(1)

    ok(f"Dataset procesado: {len(datos_proc):,} muestras")
    ok(f"Features: {FEATURES}  →  Objetivo: {TARGET}")

    # Entrenamiento
    step("Entrenando modelos supervisados (RegresionLineal + BosqueAleatorio)...")
    try:
        entrenador = EntrenadorDemanda(datos_proc, FEATURES, TARGET)
        modelos, X_test, y_test = entrenador.generar_modelos_entrenados()
    except Exception as e:
        fail(f"Error en entrenamiento: {e}")
        sys.exit(1)

    ok(f"Modelos entrenados. Datos de prueba: {len(X_test):,} muestras")

    # Evaluación
    step("Calculando métricas de evaluación...")
    try:
        evaluador = EvaluadorModelos(modelos, X_test, y_test)
        metricas = evaluador.calcular_metricas()
    except Exception as e:
        fail(f"Error en evaluación: {e}")
        sys.exit(1)

    print()
    print(f"    {'Modelo':<26} {'MAE':>10} {'MSE':>16} {'R²':>8}")
    print(f"    {'─' * 63}")
    mejor_nombre, mejor_mae = None, float("inf")
    for nombre, m in metricas.items():
        print(f"    {nombre:<26} {m['MAE']:>10.2f} {m['MSE']:>16.2f} {m['R2']:>8.4f}")
        if m["MAE"] < mejor_mae:
            mejor_mae, mejor_nombre = m["MAE"], nombre
    print()
    ok(f"Mejor modelo seleccionado: {mejor_nombre}  (MAE = {mejor_mae:.2f})")

    # Predicciones por categoría para informar al Módulo A
    step("Generando predicciones de demanda por categoría (viernes del último mes)...")
    modelo_rf = modelos[mejor_nombre]
    ultimo_mes = int(datos_proc["Mes"].max())
    categorias_unicas = datos_proc["Product_Category"].unique()

    predicciones = {}
    errores_prediccion = []
    for cat in categorias_unicas:
        try:
            cod = procesador.codificador.transform([cat])[0]
            val = modelo_rf.predict([[ultimo_mes, 4, cod]])[0]  # 4 = viernes
            predicciones[cat] = max(0.0, float(val))
        except Exception:
            errores_prediccion.append(cat)

    if errores_prediccion:
        warn(f"No se pudo predecir para: {errores_prediccion}")

    top_cat = max(predicciones, key=predicciones.get)
    ok(
        f"Categoría con mayor demanda predicha: '{top_cat}' "
        f"(Q = {predicciones[top_cat]:,.2f})"
    )

    return modelos, mejor_nombre, metricas, datos_proc, predicciones, procesador


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — Predicción Deep Learning (Módulo C)
# ═══════════════════════════════════════════════════════════════════════════════
def fase_dl():
    header("FASE 3/5 │ Predicción Deep Learning (Módulo C)")

    try:
        from dl.model import DLPredictor

        step("Inicializando red neuronal (Módulo C)...")
        predictor = DLPredictor()
        resultado = predictor.predict()
        ok(f"Predicción DL generada: {resultado}")
        return resultado

    except ImportError:
        warn("Módulo C (src/dl/model.py) no encontrado — integración pendiente.")
        warn("Interfaz esperada: clase DLPredictor con método predict() → dict")
        step("Pipeline continúa con predicciones del Módulo B como fallback.")
        return None
    except Exception as e:
        warn(f"Error en Módulo C: {e}. Continuando con fallback ML.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4 — Análisis NLP / LLM (Módulo D)
# ═══════════════════════════════════════════════════════════════════════════════
def fase_nlp(predicciones: dict, skip_nlp: bool = False):
    header("FASE 4/5 │ Análisis NLP / LLM (Módulo D)")

    nlp_results = {"summary": None, "sentimiento_ok": False}

    if skip_nlp:
        warn("--skip-nlp activo. Se omite carga del modelo BERT.")
        return nlp_results

    # 4a. Resumen automático de ventas
    try:
        from nlp.summary_generator import generate_sales_summary

        step("Generando resumen automático de reporte de ventas...")
        top_cat = max(predicciones, key=predicciones.get)
        top_val = predicciones[top_cat]
        resto = [v for k, v in predicciones.items() if k != top_cat]
        avg_resto = np.mean(resto) if resto else top_val
        crecimiento = round((top_val - avg_resto) / avg_resto * 100, 2) if avg_resto else 0.0

        payload = {
            "ventas_totales": int(top_val),
            "categoria_top": top_cat,
            "crecimiento": crecimiento,
        }
        resumen = generate_sales_summary(payload)
        ok(f'Resumen: "{resumen["summary"]}"')
        nlp_results["summary"] = resumen["summary"]

    except ImportError as e:
        warn(f"summary_generator no disponible: {e}")
    except Exception as e:
        warn(f"Error al generar resumen: {e}")

    # 4b. Análisis de sentimiento sobre reseñas de muestra
    try:
        from nlp.sentiment import analyze_sentiment

        step("Analizando sentimiento de reseñas de muestra (modelo BERT multilingüe)...")
        test_reviews = [
            ("Positivo",  "El producto llegó rápido y funciona excelente."),
            ("Negativo",  "El pedido vino dañado y el soporte nunca respondió."),
            ("Ambiguo",   "Este producto está mortal"),          # Jerga positiva → posible error
            ("Sarcástico","Excelente servicio... nunca llegó el pedido"),  # Ironía
        ]

        print()
        print(f"    {'Tipo':<12} {'Texto':<42} {'Etiqueta':>14} {'Score':>7}")
        print(f"    {'─' * 78}")
        for tipo, texto in test_reviews:
            try:
                res = analyze_sentiment(texto)
                etiqueta = res["label"]
                score = res["score"]
                texto_corto = texto[:39] + "..." if len(texto) > 39 else texto
                print(f"    {tipo:<12} {texto_corto:<42} {etiqueta:>14} {score:>7.4f}")
            except Exception as e_inner:
                warn(f"Error analizando '{texto[:30]}...': {e_inner}")

        print()
        nlp_results["sentimiento_ok"] = True
        ok("Análisis de sentimiento completado")

    except ImportError as e:
        warn(f"Módulo de sentimiento no disponible (requiere torch + transformers): {e}")
        warn("Instalar: pip install transformers torch sentencepiece")
    except Exception as e:
        warn(f"Error en análisis de sentimiento: {e}")

    return nlp_results


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 5 — Optimización de Rutas A* (Módulo A)
# ═══════════════════════════════════════════════════════════════════════════════
def fase_astar(predicciones: dict):
    header("FASE 5/5 │ Optimización de Rutas A* (Módulo A)")

    try:
        from search_csp.agent import WarehouseEnvironment, InventoryAgent
        from search_csp.algorithm import a_star_search
    except ImportError as e:
        fail(f"No se pudo importar Módulo A (search_csp): {e}")
        return []

    # Seleccionar top-3 categorías por demanda predicha para reabastecer
    top3 = sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:3]

    step("Top 3 categorías priorizadas para reabastecimiento:")
    for i, (cat, dem) in enumerate(top3, 1):
        ubic = CATEGORY_LOCATIONS.get(cat, (6, 6))
        ok(f"  #{i}  {cat:<22}  Demanda predicha = {dem:>9,.2f}  →  Estante {ubic}")

    print()
    step("Ejecutando algoritmo A* para cada parada de recolección...")

    rutas = []
    pos_actual = WAREHOUSE_ENTRY

    for cat, dem in top3:
        destino = CATEGORY_LOCATIONS.get(cat, (6, 6))
        try:
            env = WarehouseEnvironment(WAREHOUSE_GRID, pos_actual, destino)
            agente = InventoryAgent(env)
            ruta = a_star_search(agente)

            if ruta:
                pasos = len(ruta) - 1
                rutas.append((cat, destino, ruta))
                ok(
                    f"'{cat}': {pos_actual} → {destino}  |  "
                    f"{pasos} pasos  |  Costo g(n) = {pasos}"
                )
                pos_actual = destino
            else:
                warn(f"No se encontró ruta a '{cat}' desde {pos_actual}. Saltando.")

        except Exception as e:
            warn(f"Error calculando ruta para '{cat}': {e}")

    if rutas:
        distancia_total = sum(len(r) - 1 for _, _, r in rutas)
        ok(f"Recorrido completo: {distancia_total} celdas de almacén optimizadas")

    return rutas


# ═══════════════════════════════════════════════════════════════════════════════
# Análisis de Sesgos (Módulo E — Ética)
# ═══════════════════════════════════════════════════════════════════════════════
def analizar_sesgos(datos_completos: pd.DataFrame, modelos: dict, mejor_nombre: str,
                    datos_proc: pd.DataFrame, predicciones: dict):
    header("ANÁLISIS DE EQUIDAD (FAIRNESS) │ Módulo E — Ética")

    # 1. Distribución demográfica
    step("Distribución demográfica en el dataset:")
    total = len(datos_completos)

    if "Customer_Gender" in datos_completos.columns:
        print()
        print(f"    {'Género':<22} {'Registros':>10} {'%':>8}")
        print(f"    {'─' * 44}")
        for genero, cnt in datos_completos["Customer_Gender"].value_counts(dropna=False).items():
            label = str(genero) if pd.notna(genero) else "Sin registro (NaN)"
            print(f"    {label:<22} {cnt:>10,} {cnt/total*100:>8.1f}%")

    # 2. Sesgo de representación por categoría y género
    if "Customer_Gender" in datos_completos.columns and "Product_Category" in datos_completos.columns:
        print()
        step("Distribución de género por categoría (% de transacciones):")

        cat_gender = (
            datos_completos.groupby(["Product_Category", "Customer_Gender"])
            .size()
            .unstack(fill_value=0)
        )
        cat_gender_pct = cat_gender.div(cat_gender.sum(axis=1), axis=0) * 100

        cols_mostrar = [c for c in ["Female", "Male", "Other"] if c in cat_gender_pct.columns]
        header_row = f"    {'Categoría':<22}" + "".join(f"{c:>10}" for c in cols_mostrar) + f"  {'Dominancia'}"
        print()
        print(header_row)
        print(f"    {'─' * 72}")

        cat_sesgo = {}
        for cat in sorted(cat_gender_pct.index):
            valores = [cat_gender_pct.loc[cat, c] if c in cat_gender_pct.columns else 0 for c in cols_mostrar]
            max_pct = max(valores)
            gen_dom = cols_mostrar[valores.index(max_pct)] if valores else "N/A"
            sesgo_label = "ALTO" if max_pct > 38 else "MEDIO" if max_pct > 34 else "BAJO"
            cat_sesgo[cat] = {"gen_dom": gen_dom, "pct_dom": max_pct, "nivel": sesgo_label}
            fila = f"    {cat:<22}" + "".join(f"{v:>10.1f}" for v in valores)
            print(f"{fila}  {gen_dom} {max_pct:.1f}% [{sesgo_label}]")

    # 3. MAE por categoría con sesgo demográfico
    print()
    step("Evaluando precisión del modelo por grupo de sesgo demográfico...")

    modelo = modelos[mejor_nombre]
    X_all = datos_proc[FEATURES]
    y_all = datos_proc[TARGET]
    y_pred_all = modelo.predict(X_all)
    datos_proc_eval = datos_proc.copy()
    datos_proc_eval["y_pred"] = y_pred_all
    datos_proc_eval["error_abs"] = np.abs(y_all.values - y_pred_all)

    # MAE global
    mae_global = datos_proc_eval["error_abs"].mean()
    ok(f"MAE global del modelo ({mejor_nombre}): {mae_global:.2f}")

    # MAE por categoría
    mae_por_cat = datos_proc_eval.groupby("Product_Category")["error_abs"].mean().sort_values(ascending=False)
    print()
    print(f"    {'Categoría':<22} {'MAE':>10}  {'vs. Global':>12}")
    print(f"    {'─' * 48}")
    for cat, mae_cat in mae_por_cat.items():
        diff_pct = (mae_cat - mae_global) / mae_global * 100
        signo = "+" if diff_pct >= 0 else ""
        print(f"    {cat:<22} {mae_cat:>10.2f}  {signo}{diff_pct:>10.1f}%")

    # 4. Sesgo etario
    if "Customer_Age" in datos_completos.columns:
        print()
        step("Distribución de ventas por grupo etario:")
        dc = datos_completos.copy()
        dc["Grupo_Edad"] = pd.cut(
            dc["Customer_Age"],
            bins=[0, 25, 40, 60, 120],
            labels=["18-25", "26-40", "41-60", "60+"],
        )
        if "Sales_Amount" in dc.columns:
            tabla_edad = dc.groupby("Grupo_Edad", observed=True)["Sales_Amount"].agg(["mean", "count", "std"])
            print()
            print(f"    {'Grupo':<10} {'Venta Promedio':>16} {'Desv. Estándar':>16} {'Registros':>12}")
            print(f"    {'─' * 58}")
            for grupo, row in tabla_edad.iterrows():
                print(
                    f"    {str(grupo):<10} {row['mean']:>16.2f} {row['std']:>16.2f} {int(row['count']):>12,}"
                )

    ok("Análisis de equidad finalizado. Detalles completos: docs/ethics_analysis.md")


# ═══════════════════════════════════════════════════════════════════════════════
# Reporte Final
# ═══════════════════════════════════════════════════════════════════════════════
def reporte_final(metricas: dict, mejor_nombre: str, predicciones: dict,
                  rutas: list, nlp_results: dict, dl_result):
    header("REPORTE EJECUTIVO │ Sistema de Optimización de Inventario")

    top_cat = max(predicciones, key=predicciones.get) if predicciones else "N/A"
    top_val = predicciones.get(top_cat, 0)

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │          RESUMEN EJECUTIVO — PIPELINE COMPLETO                   │
  └──────────────────────────────────────────────────────────────────┘

  [DATOS]      100,000 transacciones de retail | {len(predicciones)} categorías procesadas

  [MÓDULO B]   Modelo: {mejor_nombre}
               MAE  = {metricas[mejor_nombre]['MAE']:.2f}   MSE  = {metricas[mejor_nombre]['MSE']:.2f}
               R²   = {metricas[mejor_nombre]['R2']:.4f}   (varianza de demanda explicada)

  [MÓDULO C]   {('Predicción DL: ' + str(dl_result)) if dl_result else 'Integración pendiente (agregar src/dl/model.py)'}

  [MÓDULO D]   Resumen: {(nlp_results['summary'][:75] + '...') if nlp_results.get('summary') and len(nlp_results['summary']) > 75 else nlp_results.get('summary', 'No disponible')}
               Sentimiento: {'Completado ✓' if nlp_results.get('sentimiento_ok') else 'Omitido (requiere torch/transformers)'}

  [MÓDULO A]   Categoría de mayor demanda: {top_cat} (Q={top_val:,.2f})
               Ruta de reabastecimiento A*:""")

    if rutas:
        for i, (cat, dest, ruta) in enumerate(rutas, 1):
            print(f"               Parada {i}: {cat:<22} → estante {dest}  ({len(ruta)-1} pasos)")
        total = sum(len(r) - 1 for _, _, r in rutas)
        print(f"               Distancia total optimizada: {total} celdas")
    else:
        print("               No se generaron rutas (verificar Módulo A)")

    print(f"""
  ─────────────────────────────────────────────────────────────────
  Pipeline ejecutado correctamente.
  Análisis ético completo: docs/ethics_analysis.md
  ─────────────────────────────────────────────────────────────────
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de integración — Sistema de Optimización de Inventario"
    )
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Omite la carga del modelo BERT (útil en entornos sin GPU o sin transformers)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  SISTEMA DE OPTIMIZACIÓN INTELIGENTE DE INVENTARIO")
    print("  Pipeline de Integración Completo — Módulo E")
    print("=" * 65)

    # ── Ejecutar fases en secuencia ──────────────────────────────────────────
    datos_raw, datos_completos = fase_carga_datos()

    modelos, mejor_nombre, metricas, datos_proc, predicciones, procesador = fase_ml(datos_raw)

    dl_result = fase_dl()

    nlp_results = fase_nlp(predicciones, skip_nlp=args.skip_nlp)

    rutas = fase_astar(predicciones)

    analizar_sesgos(datos_completos, modelos, mejor_nombre, datos_proc, predicciones)

    reporte_final(metricas, mejor_nombre, predicciones, rutas, nlp_results, dl_result)


if __name__ == "__main__":
    main()
