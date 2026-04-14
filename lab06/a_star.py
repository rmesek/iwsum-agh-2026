import copy
import heapq
import time


# ---------------------------------------------------------
# Node Structure
# ---------------------------------------------------------
class Node:
    def __init__(self, state, g, h, parent=None):
        self.state = state  # variable array grid [N][N]
        self.g = g  # cost from start
        self.h = h  # heuristic value
        self.f = g + h  # f(n) = g(n) + h(n)
        self.parent = parent  # pointer parent

    # Required for the priority queue to order by f(n)
    def __lt__(self, other):
        return self.f < other.f


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_blank_pos(state):
    """Finds the (row, col) of the blank tile (represented by 0)."""
    for r in range(len(state)):
        for c in range(len(state[0])):
            if state[r][c] == 0:
                return r, c
    return None


def get_target_pos(goal_state, tile_value):
    """Finds the target (row, col) of a specific tile in the goal state."""
    for r in range(len(goal_state)):
        for c in range(len(goal_state[0])):
            if goal_state[r][c] == tile_value:
                return r, c
    return None


def state_to_tuple(state):
    """Converts a 2D list to a tuple of tuples for hashing in closed sets."""
    return tuple(tuple(row) for row in state)


# ---------------------------------------------------------
# Algorithm 3: Generate Neighbors
# ---------------------------------------------------------
def generate_neighbors(state):
    neighbors = []
    r, c = get_blank_pos(state)

    # Moves: UP (-1, 0), DOWN (1, 0), LEFT (0, -1), RIGHT (0, 1)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n = len(state)

    for dr, dc in moves:
        r_prime, c_prime = r + dr, c + dc
        # If inside board
        if 0 <= r_prime < n and 0 <= c_prime < n:
            new_state = copy.deepcopy(state)
            # Swap blank with tile
            new_state[r][c], new_state[r_prime][c_prime] = (
                new_state[r_prime][c_prime],
                new_state[r][c],
            )
            neighbors.append(new_state)

    return neighbors


# ---------------------------------------------------------
# Heuristic 1: Hamming Distance (Misplaced Tiles)
# ---------------------------------------------------------
def heuristic_hamming(state, goal_state):
    misplaced = 0
    n = len(state)
    for r in range(n):
        for c in range(n):
            tile = state[r][c]
            # Do not count the blank tile
            if tile != 0 and tile != goal_state[r][c]:
                misplaced += 1
    return misplaced


# ---------------------------------------------------------
# Heuristic 2: Manhattan Distance
# ---------------------------------------------------------
def heuristic_manhattan(state, goal_state):
    manhattan_dist = 0
    n = len(state)
    for r in range(n):
        for c in range(n):
            tile = state[r][c]
            if tile != 0:
                r_g, c_g = get_target_pos(goal_state, tile)
                manhattan_dist += abs(r - r_g) + abs(c - c_g)
    return manhattan_dist


# ---------------------------------------------------------
# Algorithm 2: Manhattan + Linear Conflict
# ---------------------------------------------------------
def heuristic_linear_conflict(state, goal_state):
    manhattan_dist = heuristic_manhattan(state, goal_state)
    linear_conflict = 0
    n = len(state)

    # Check conflicts in rows
    for r in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                t_i = state[r][i]
                t_j = state[r][j]
                if t_i != 0 and t_j != 0:
                    rg_i, cg_i = get_target_pos(goal_state, t_i)
                    rg_j, cg_j = get_target_pos(goal_state, t_j)
                    # If both belong in this row in goal AND are in wrong order
                    if rg_i == r and rg_j == r and cg_i > cg_j:
                        linear_conflict += 2

    # Check conflicts in columns
    for c in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                t_i = state[i][c]
                t_j = state[j][c]
                if t_i != 0 and t_j != 0:
                    rg_i, cg_i = get_target_pos(goal_state, t_i)
                    rg_j, cg_j = get_target_pos(goal_state, t_j)
                    # If both belong in this col in goal AND are in wrong order
                    if cg_i == c and cg_j == c and rg_i > rg_j:
                        linear_conflict += 2

    return manhattan_dist + linear_conflict


# ---------------------------------------------------------
# Algorithm 1: A* Search
# ---------------------------------------------------------
def astar(start_state, goal_state, heuristic_func):
    # Initialize start node
    start_g = 0
    start_h = heuristic_func(start_state, goal_state)
    start_node = Node(start_state, start_g, start_h)

    # Priority queue ordered by f(n)
    open_list = []
    heapq.heappush(open_list, start_node)

    # Tracking visited states to prevent loops
    closed_set = set()

    # Keep track of the best g for a state to avoid duplicate inferior nodes
    best_g = {state_to_tuple(start_state): start_g}

    nodes_evaluated = 0

    while open_list:
        # Node with lowest f
        current = heapq.heappop(open_list)
        nodes_evaluated += 1

        # If current = goal then return RECONSTRUCTPATH
        if current.state == goal_state:
            path = []
            temp = current
            while temp is not None:
                path.append(temp.state)
                temp = temp.parent
            return path[::-1], nodes_evaluated  # Return reversed path and cost

        current_tuple = state_to_tuple(current.state)
        closed_set.add(current_tuple)

        # for all neighbor in GENERATENEIGHBORS
        for neighbor_state in generate_neighbors(current.state):
            neighbor_tuple = state_to_tuple(neighbor_state)

            if neighbor_tuple in closed_set:
                continue

            tentative_g = current.g + 1

            # if neighbor not in open OR tentative_g < g(neighbor)
            if neighbor_tuple not in best_g or tentative_g < best_g[neighbor_tuple]:
                best_g[neighbor_tuple] = tentative_g
                h = heuristic_func(neighbor_state, goal_state)
                neighbor_node = Node(neighbor_state, tentative_g, h, current)
                heapq.heappush(open_list, neighbor_node)

    return None, nodes_evaluated  # Return failure


# ---------------------------------------------------------
# Testing & Cost Comparison (N=8)
# ---------------------------------------------------------
if __name__ == "__main__":
    # N=8 puzzle (3x3 grid)
    # Goal state (standard 1-8 arrangement with 0 as blank)
    goal_state = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0],
    ]

    # A solvable start state (e.g., requires ~14 moves)
    start_state = [
        [1, 2, 3],
        [4, 0, 5],
        [7, 8, 6],
    ]

    print("--- 8-Puzzle A* Heuristic Comparison ---")

    heuristics = [
        ("Hamming (Misplaced Tiles)", heuristic_hamming),
        ("Manhattan Distance", heuristic_manhattan),
        ("Manhattan + Linear Conflict", heuristic_linear_conflict),
    ]

    for name, func in heuristics:
        print(f"\nRunning with {name}...")
        start_time = time.time()

        path, nodes_eval = astar(start_state, goal_state, func)

        elapsed_time = time.time() - start_time

        if path:
            print(f"Solution found in {len(path) - 1} moves.")
            print(f"Nodes Evaluated (Cost of Search): {nodes_eval}")
            print(f"Time Taken: {elapsed_time:.4f} seconds")
        else:
            print("No solution found.")
