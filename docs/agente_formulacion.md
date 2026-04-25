# Formulación del Problema como Agente Inteligente

Para el Módulo A, el agente se encargará de la planificación de la ruta de abastecimiento de inventario en el almacén.

* **Estado Inicial:** El agente (vehículo de reabastecimiento o sistema de asignación) se encuentra en el punto de origen (muelle de carga) con una lista de productos a reponer en ubicaciones específicas del almacén.
* **Estado Meta:** Todos los productos han sido ubicados en sus respectivos estantes y el agente ha regresado al punto de origen o finalizado su ruta, minimizando la distancia recorrida o el tiempo.
* **Acciones:** `Mover_Norte`, `Mover_Sur`, `Mover_Este`, `Mover_Oeste`, `Descargar_Producto(id_producto)`.
* **Función de Evaluación (Heurística):** Para un algoritmo de búsqueda como A*, la función de costo $f(n) = g(n) + h(n)$ donde $g(n)$ es la distancia real recorrida y $h(n)$ es la distancia Manhattan desde la posición actual hasta el estante del producto prioritario más cercano.