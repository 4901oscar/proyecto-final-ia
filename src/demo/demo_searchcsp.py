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
