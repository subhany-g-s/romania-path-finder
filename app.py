from flask import Flask, render_template, request, jsonify
from collections import deque
import heapq

# ---------------- Romania Map ---------------- #
romania_map = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Pitesti': 138, 'Rimnicu Vilcea': 146},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

# Straight-line heuristic (H(n) to Bucharest)
heuristic = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
    'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
    'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193,
    'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80,
    'Vaslui': 199, 'Zerind': 374
}

# Real-world coordinates of cities (latitude, longitude)
city_coords = {
    'Arad': (46.1667, 21.3167),
    'Zerind': (46.6167, 21.5167),
    'Oradea': (47.0667, 21.9333),
    'Sibiu': (45.7928, 24.1522),
    'Timisoara': (45.7597, 21.23),
    'Lugoj': (45.6886, 21.9031),
    'Mehadia': (44.9000, 22.3667),
    'Drobeta': (44.6369, 22.6597),
    'Craiova': (44.3167, 23.8),
    'Rimnicu Vilcea': (45.1, 24.3667),
    'Fagaras': (45.8416, 24.9741),
    'Pitesti': (44.8565, 24.8692),
    'Bucharest': (44.4268, 26.1025),
    'Giurgiu': (43.9, 25.9667),
    'Urziceni': (44.7167, 26.6333),
    'Hirsova': (44.6833, 27.95),
    'Eforie': (44.0667, 28.6333),
    'Vaslui': (46.6333, 27.7333),
    'Iasi': (47.1667, 27.6),
    'Neamt': (46.9167, 26.3333)
}

# ---------------- Search Algorithms ---------------- #
def bfs_all_paths(graph, start, goal):
    queue = deque([[start]])
    all_paths = []
    explored = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        explored.add(node)
        if len(path) > len(graph) + 1:
            continue
        if node == goal:
            all_paths.append(path)
            continue
        for neighbor in sorted(graph[node].keys()):
            if neighbor not in path:
                queue.append(path + [neighbor])
                explored.add(neighbor)
    return all_paths, list(explored)

def dfs_all_paths(graph, start, goal):
    stack = [[start]]
    all_paths = []
    explored = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        explored.add(node)
        if len(path) > len(graph) + 1:
            continue
        if node == goal:
            all_paths.append(path)
            continue
        for neighbor in sorted(graph[node].keys(), reverse=True):
            if neighbor not in path:
                stack.append(path + [neighbor])
                explored.add(neighbor)
    return all_paths, list(explored)

def astar_search(graph, start, goal):
    open_set = [(heuristic.get(start, 0), 0, [start])]
    min_cost_to_reach = {start: 0}
    explored = set()
    while open_set:
        f, g, path = heapq.heappop(open_set)
        node = path[-1]
        explored.add(node)
        if node == goal:
            return [path], list(explored)
        if g > min_cost_to_reach.get(node, float('inf')):
            continue
        for neighbor, cost in graph[node].items():
            new_g = g + cost
            new_f = new_g + heuristic.get(neighbor, 0)
            if new_g < min_cost_to_reach.get(neighbor, float('inf')):
                min_cost_to_reach[neighbor] = new_g
                heapq.heappush(open_set, (new_f, new_g, path + [neighbor]))
                explored.add(neighbor)
    return [], list(explored)

def uniform_cost_search(graph, start, goal):
    pq = [(0, [start])]
    min_cost_to_reach = {start: 0}
    explored = set()
    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]
        explored.add(node)
        if node == goal:
            return [path], list(explored)
        if cost > min_cost_to_reach.get(node, float('inf')):
            continue
        for neighbor, edge_cost in graph[node].items():
            new_cost = cost + edge_cost
            if new_cost < min_cost_to_reach.get(neighbor, float('inf')):
                min_cost_to_reach[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, path + [neighbor]))
                explored.add(neighbor)
    return [], list(explored)

def greedy_best_first_search(graph, start, goal):
    open_set = [(heuristic.get(start, 0), [start])]
    visited = set()
    explored = set()
    while open_set:
        h, path = heapq.heappop(open_set)
        node = path[-1]
        explored.add(node)
        if node == goal:
            return [path], list(explored)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in sorted(graph[node].keys()):
            if neighbor not in visited:
                heapq.heappush(open_set, (heuristic.get(neighbor, 0), path + [neighbor]))
                explored.add(neighbor)
    return [], list(explored)

def depth_limited_search(graph, start, goal, limit):
    explored = set()
    def recursive_dls(path, node, limit):
        explored.add(node)
        if node == goal:
            return [path]
        if limit <= 0:
            return []
        all_paths = []
        for neighbor in sorted(graph[node].keys()):
            if neighbor not in path:
                explored.add(neighbor)
                all_paths.extend(recursive_dls(path + [neighbor], neighbor, limit - 1))
        return all_paths
    paths = recursive_dls([start], start, limit)
    return paths, list(explored)

def ao_star_search(graph, start, goal):
    return astar_search(graph, start, goal)

def get_path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        city1, city2 = path[i], path[i + 1]
        cost += romania_map.get(city1, {}).get(city2, float('inf'))
    return cost

# ---------------- Flask App ---------------- #
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize all variables with default values
    result = None
    path_coords = []
    explored_coords = []
    cities_list = sorted(list(romania_map.keys()))
    algorithm_name = ""
    path_cost = 0
    num_cities = 0
    all_city_coords = city_coords

    if request.method == "POST":
        start = request.form["start"]
        goal = request.form["goal"]
        algo = request.form["algo"]

        option = request.form.get("option", "lowest") if algo in ["BFS", "DFS"] else "single"
        try:
            depth_limit = int(request.form.get("limit", 4))
        except:
            depth_limit = 4

        if start not in romania_map or goal not in romania_map:
            result = f"Invalid city name(s): {start} or {goal}."
        else:
            all_paths = []
            explored = []
            
            if algo == "BFS":
                all_paths, explored = bfs_all_paths(romania_map, start, goal)
                algorithm_name = "Breadth-First Search"
            elif algo == "DFS":
                all_paths, explored = dfs_all_paths(romania_map, start, goal)
                algorithm_name = "Depth-First Search"
            elif algo == "ASTAR":
                all_paths, explored = astar_search(romania_map, start, goal)
                algorithm_name = "A* Search"
            elif algo == "UCS":
                all_paths, explored = uniform_cost_search(romania_map, start, goal)
                algorithm_name = "Uniform Cost Search"
            elif algo == "GREEDY":
                all_paths, explored = greedy_best_first_search(romania_map, start, goal)
                algorithm_name = "Greedy Best-First Search"
            elif algo == "DLS":
                all_paths, explored = depth_limited_search(romania_map, start, goal, depth_limit)
                algorithm_name = f"Depth-Limited Search (Limit: {depth_limit})"
            elif algo == "AO*":
                all_paths, explored = ao_star_search(romania_map, start, goal)
                algorithm_name = "AO* Search"

            if all_paths:
                # Determine which path to display
                if algo in ["BFS", "DFS"]:
                    if option == "all":
                        output = []
                        for i, path in enumerate(all_paths, 1):
                            steps = " → ".join(path)
                            cost = get_path_cost(path)
                            output.append(f"<div class='mb-2'><strong>Path {i}:</strong> {steps} <span class='text-gray-600'>| Cost: {cost} | Steps: {len(path)-1}</span></div>")
                        result = "".join(output)
                        p = all_paths[0]  # for visualization, pick first path
                    elif option == "shortest":
                        p = min(all_paths, key=len)
                        result = f"<strong>Shortest Path:</strong> {' → '.join(p)} <span class='text-gray-600'>(Cost: {get_path_cost(p)}, Steps: {len(p)-1})</span>"
                    else:
                        p = min(all_paths, key=get_path_cost)
                        result = f"<strong>Lowest Cost Path:</strong> {' → '.join(p)} <span class='text-gray-600'>(Cost: {get_path_cost(p)}, Steps: {len(p)-1})</span>"
                else:
                    p = min(all_paths, key=get_path_cost)
                    result = f"<strong>Optimal Path:</strong> {' → '.join(p)}"

                # Extract coordinates for visualization
                path_coords = [city_coords[c] for c in p if c in city_coords]
                explored_coords = [city_coords[c] for c in explored if c in city_coords and c not in p]
                path_cost = get_path_cost(p)
                num_cities = len(p)
            else:
                result = f"No path found from {start} to {goal}."

    return render_template("index.html", 
                         result=result, 
                         cities=cities_list, 
                         path_coords=path_coords,
                         explored_coords=explored_coords,
                         algorithm_name=algorithm_name,
                         path_cost=path_cost,
                         num_cities=num_cities,
                         all_city_coords=city_coords)

if __name__ == "__main__":
    app.run(debug=True)