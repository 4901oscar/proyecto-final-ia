from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class EvaluadorModelos:
    def __init__(self, modelos: dict, X_prueba, y_prueba):
        self.modelos = modelos
        self.X_prueba = X_prueba
        self.y_prueba = y_prueba

    def calcular_metricas(self) -> dict:
        resultados = {}
        for nombre, modelo in self.modelos.items():
            try:
                # Modelos DL de series temporales con evaluación propia
                if hasattr(modelo, 'obtener_predicciones_y_reales'):
                    predicciones, y_reales = modelo.obtener_predicciones_y_reales()
                # Manejar modelos de DL tabulares (que tienen método predecir)
                elif hasattr(modelo, 'predecir'):
                    predicciones = modelo.predecir(self.X_prueba)
                    y_reales = self.y_prueba
                else:
                    predicciones = modelo.predict(self.X_prueba)
                    y_reales = self.y_prueba
                
                resultados[nombre] = {
                    'MSE': mean_squared_error(y_reales, predicciones),
                    'MAE': mean_absolute_error(y_reales, predicciones),
                    'RMSE': np.sqrt(mean_squared_error(y_reales, predicciones)),
                    'R2': r2_score(y_reales, predicciones)
                }
            except Exception as e:
                print(f"Error evaluando {nombre}: {str(e)}")
                resultados[nombre] = {'error': str(e)}
        
        return resultados
    
    def generar_reporte(self) -> str:
        """Genera un reporte comparativo de todos los modelos"""
        metricas = self.calcular_metricas()
        
        reporte = "\n" + "="*70 + "\n"
        reporte += "REPORTE DE EVALUACIÓN DE MODELOS\n"
        reporte += "="*70 + "\n\n"
        
        for nombre, valores in metricas.items():
            reporte += f"{nombre}:\n"
            if 'error' in valores:
                reporte += f"  Error: {valores['error']}\n"
            else:
                for metrica, valor in valores.items():
                    reporte += f"  {metrica}: {valor:.4f}\n"
            reporte += "\n"
        
        return reporte