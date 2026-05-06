from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EvaluadorModelos:
    def __init__(self, modelos: dict, X_prueba, y_prueba):
        self.modelos = modelos
        self.X_prueba = X_prueba
        self.y_prueba = y_prueba

    def calcular_metricas(self) -> dict:
        resultados = {}
        for nombre, modelo in self.modelos.items():
            predicciones = modelo.predict(self.X_prueba)
            resultados[nombre] = {
                'MSE': mean_squared_error(self.y_prueba, predicciones),
                'MAE': mean_absolute_error(self.y_prueba, predicciones),
                'R2': r2_score(self.y_prueba, predicciones)
            }
        return resultados