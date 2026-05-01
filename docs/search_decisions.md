# Justificación de Decisiones: Módulo de Búsqueda (Módulo A)

## 1. Algoritmo Elegido: Búsqueda A* (A-Star)
Para el problema de optimización de rutas de recolección de inventario en el almacén, se implementó el algoritmo A*.

### Justificación de la Heurística
Se utilizó la distancia Manhattan como función heurística $h(n)$. Dado que los montacargas y personal en los pasillos de un almacén estándar solo pueden moverse en ángulos rectos (Norte, Sur, Este, Oeste), la distancia Manhattan representa de forma exacta el costo mínimo teórico para alcanzar el objetivo. Esta heurística es **admisible** (nunca sobreestima el costo real) y **consistente**, garantizando que A* encuentre la solución óptima.

### Complejidad
* **Complejidad de Tiempo:** En el peor de los casos, la complejidad es $O(b^d)$, donde $b$ es el factor de ramificación (máximo 4 movimientos posibles) y $d$ es la profundidad de la solución. Sin embargo, gracias a la heurística de Manhattan, el espacio de búsqueda se poda significativamente en la dirección del objetivo.
* **Complejidad de Espacio:** $O(b^d)$, ya que mantiene todos los nodos generados en memoria dentro de las estructuras `frontier` y `came_from`.

## 2. Comparación con Alternativa Descartada: Búsqueda en Anchura (BFS)
Inicialmente se consideró BFS (Breadth-First Search). BFS garantiza encontrar el camino más corto en grafos sin pesos, lo cual aplica a nuestro grid. 
**Razón del descarte:** BFS explora el espacio de manera radial y exhaustiva en todas las direcciones simultáneamente. En un entorno de almacén a gran escala, esto implicaría evaluar miles de pasillos irrelevantes que están en la dirección opuesta al producto buscado, agotando recursos. A*, al estar guiado por su función $f(n) = g(n) + h(n)$, prioriza la expansión de los nodos que se acercan físicamente al punto de recolección, reduciendo drásticamente el tiempo de ejecución y el consumo de memoria en comparación con BFS.