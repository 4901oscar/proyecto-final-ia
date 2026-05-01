class WarehouseEnvironment:
    def __init__(self, grid, start, goal):
        """
        grid: Matriz 2D donde 0 es pasillo libre y 1 es estantería (obstáculo).
        start: Tupla (x, y) con la posición inicial.
        goal: Tupla (x, y) con la ubicación del producto a recolectar.
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def is_valid_state(self, state):
        x, y = state
        # Verifica límites del grid y que no sea un obstáculo
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return self.grid[x][y] == 0
        return False

    def get_successors(self, state):
        x, y = state
        successors = []
        # Movimientos: Arriba, Abajo, Izquierda, Derecha
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in moves:
            next_state = (x + dx, y + dy)
            if self.is_valid_state(next_state):
                # El costo de moverse un bloque es siempre 1
                successors.append((next_state, 1)) 
        return successors

class InventoryAgent:
    def __init__(self, environment):
        self.env = environment

    def get_start_state(self):
        return self.env.start

    def is_goal(self, state):
        return state == self.env.goal