# Módulo `search_csp`

Descripción
-----------
El módulo `search_csp` contiene la implementación del entorno de almacén y del agente que utiliza el algoritmo A* para planificar rutas óptimas de recolección en un grid (mapa del almacén). Está pensado para integrarse con el pipeline de integración (`src/integration/pipeline.py`) como el Módulo A — Optimización de rutas.

Estructura
----------
- `src/search_csp/agent.py`
  - `WarehouseEnvironment(grid, start, goal)`: representa el entorno del almacén.
  - `InventoryAgent(environment)`: clase agente que expone `get_start_state()` e `is_goal(state)` para el algoritmo de búsqueda.
- `src/search_csp/algorithm.py`
  - `a_star_search(agent)`: implementación del algoritmo A* que recibe un `agent` con la API mínima requerida.

Cómo funciona
--------------
1. Representación del almacén
   - El almacén se modela como una matriz 2D (`grid`) donde `0` indica pasillo libre y `1` indica estantería/obstáculo.
   - Las coordenadas se usan como tuplas `(fila, columna)`.

2. Entorno y agente
   - `WarehouseEnvironment` valida estados (`is_valid_state`) y genera sucesores ortogonales (arriba, abajo, izquierda, derecha) con coste unitario.
   - `InventoryAgent` encapsula el entorno y ofrece `get_start_state()` y `is_goal(state)` para el algoritmo de búsqueda.

3. Algoritmo A*
   - `a_star_search` implementa una cola de prioridad (heap) con prioridad `f(n) = g(n) + h(n)`.
   - Heurística: distancia Manhattan `h(n)` entre el estado actual y el objetivo (admisible para movimientos 4-direccionales).
   - Devuelve la ruta (lista de tuplas) del inicio a la meta o `None` si no existe camino.

API e integración con el pipeline
---------------------------------
- Interfaz mínima esperada por `src/integration/pipeline.py`:
  - `from search_csp.agent import WarehouseEnvironment, InventoryAgent`
  - `from search_csp.algorithm import a_star_search`

- Flujo típico en el pipeline:
  1. Seleccionar top-N categorías con mayor demanda predicha.
  2. Para cada categoría, obtener la posición del estante desde `CATEGORY_LOCATIONS`.
  3. Inicializar `WarehouseEnvironment(WAREHOUSE_GRID, pos_actual, destino)` y `InventoryAgent(env)`.
  4. Llamar a `a_star_search(agente)` y procesar la ruta retornada.

Ejemplo mínimo
--------------
```python
from search_csp.agent import WarehouseEnvironment, InventoryAgent
from search_csp.algorithm import a_star_search

grid = [
    [0,0,0],
    [0,1,0],
    [0,0,0]
]
start = (2, 0)
goal = (0, 2)

env = WarehouseEnvironment(grid, start, goal)
agent = InventoryAgent(env)
route = a_star_search(agent)
print(route)  # [(2,0), (1,0), (0,0), (0,1), (0,2)]
```

Notas y recomendaciones
----------------------
- Asegurar que `CATEGORY_LOCATIONS` usa coordenadas compatibles con la matriz `WAREHOUSE_GRID` (fila, columna).
- Revisar espacios no transitables (1) para evitar rutas imposibles; el pipeline ya maneja rutas inexistentes con una advertencia y sigue con la siguiente parada.
- Si se desea optimizar múltiples paradas (TSP-like), implementar un solver adicional o heurística de greedy + A*.

Referencias internas
-------------------
- Pipeline de integración: `src/integration/pipeline.py`
- Implementación del entorno: `src/search_csp/agent.py`
- Implementación del A*: `src/search_csp/algorithm.py`


Ejemplos prácticos en este proyecto
----------------------------------
1) Fragmento de integración (similar a `src/integration/pipeline.py`)

```python
# Ejemplo mínimo que replica cómo el pipeline usa el módulo
from search_csp.agent import WarehouseEnvironment, InventoryAgent
from search_csp.algorithm import a_star_search

# Definir (o importar) el grid y las ubicaciones (fila, columna)
WAREHOUSE_GRID = [
  [0,0,0,0,0,0,0,0],
  [0,1,1,0,1,1,0,0],
  [0,0,0,0,0,0,0,0],
  [0,1,1,0,1,1,0,0],
  [0,0,0,0,0,0,0,0],
  [0,1,1,0,1,1,0,0],
  [0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0],
]

CATEGORY_LOCATIONS = {
  "Toys": (1, 7),
  "Tools": (5, 1),
  "Sports": (6, 6),
}

WAREHOUSE_ENTRY = (7, 0)

# Supongamos que esto viene del Módulo B (predicciones)
predicciones = {"Toys": 8211.78, "Tools": 8210.71, "Sports": 8209.64}

pos_actual = WAREHOUSE_ENTRY
rutas = []

for cat, dem in sorted(predicciones.items(), key=lambda x: x[1], reverse=True)[:3]:
  destino = CATEGORY_LOCATIONS.get(cat, (6, 6))
  env = WarehouseEnvironment(WAREHOUSE_GRID, pos_actual, destino)
  agente = InventoryAgent(env)
  ruta = a_star_search(agente)

  if ruta:
    rutas.append((cat, destino, ruta))
    print(f"Ruta a {cat}: {ruta} ({len(ruta)-1} pasos)")
    pos_actual = destino
  else:
    print(f"No se encontró ruta a {cat} desde {pos_actual}.")

print("Rutas generadas:\n", rutas)
```

2) Script de prueba independiente

Guarda el siguiente script como `src/demo/demo_searchcsp.py` y ejecútalo para comprobar el A* en aislamiento.

```python
# src/demo/demo_searchcsp.py
from search_csp.agent import WarehouseEnvironment, InventoryAgent
from search_csp.algorithm import a_star_search

GRID = [
  [0,0,0,0],
  [0,1,1,0],
  [0,0,0,0],
]

def main():
  start = (2,0)
  goal = (0,3)
  env = WarehouseEnvironment(GRID, start, goal)
  agent = InventoryAgent(env)
  route = a_star_search(agent)
  if route:
    print(f"Ruta encontrada: {route} -> {len(route)-1} pasos")
  else:
    print("No hay ruta posible")

if __name__ == '__main__':
  main()
```

3) Casos comunes y pruebas rápidas

- Ruta inexistente: cambia `goal` a una celda que esté rodeada por `1` para verificar que el pipeline captura el caso y continúa.
- Validación de coordenadas: asegúrate de que `CATEGORY_LOCATIONS` use el mismo sistema (fila, columna) que el `WAREHOUSE_GRID`.
- Multi-parada: para optimizar varias paradas, el pipeline aplica A* de forma secuencial actualizando `pos_actual`. Para optimizar globalmente (orden óptimo de paradas) se requiere un algoritmo TSP/heurística adicional.

Con esto tendrás ejemplos ejecutables dentro del proyecto para entender exactamente cómo interactúa `search_csp` con el pipeline.

