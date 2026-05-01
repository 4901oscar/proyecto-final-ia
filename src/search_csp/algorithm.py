import heapq

def manhattan_distance(state, goal):
    """
    Función heurística h(n): Calcula la distancia Manhattan entre dos puntos.
    Es admisible porque nunca sobreestima la distancia real en un grid que solo permite movimientos ortogonales.
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def a_star_search(agent):
    """
    Implementación del algoritmo A* usando f(n) = g(n) + h(n)
    """
    start_state = agent.get_start_state()
    goal_state = agent.env.goal
    
    # Priority Queue: almacena tuplas (f_score, state)
    frontier = []
    heapq.heappush(frontier, (0, start_state))
    
    # Diccionarios para rastrear el camino y el costo real g(n)
    came_from = {start_state: None}
    cost_so_far = {start_state: 0}
    
    while frontier:
        current_f, current_state = heapq.heappop(frontier)
        
        if agent.is_goal(current_state):
            # Reconstruir el camino si llegamos a la meta
            path = []
            while current_state is not None:
                path.append(current_state)
                current_state = came_from[current_state]
            path.reverse()
            return path
            
        for next_state, cost in agent.env.get_successors(current_state):
            new_cost = cost_so_far[current_state] + cost
            
            # Si no hemos visitado el estado o encontramos un camino más barato
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + manhattan_distance(next_state, goal_state)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state
                
    return None # Retorna None si no hay camino posible