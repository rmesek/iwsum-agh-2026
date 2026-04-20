import heapq


class Node:
    def __init__(self, state, g, h, parent=None):
        self.state = state  # variable array grid [N][N]
        self.g = g  # cost from start
        self.h = h  # heuristic value
        self.parent = parent  # pointer parent

    @property
    def f(self):
        return self.g + self.h  # f(n) = g(n) + h(n)

    # required for the priority queue
    def __lt__(self, other):
        return self.f < other.f


def get_position(state, tile_value):
    """Finds the (row, col) of the specified tile value."""
    n = len(state)
    for r in range(n):
        for c in range(n):
            if state[r][c] == tile_value:
                return r, c
    return None


def state_to_tuple(state):
    """Converts a 2D list to a tuple of tuples for hashing."""
    return tuple(tuple(row) for row in state)


def generate_neighbors(state):
    neighbors = []
    r, c = get_position(state, 0)  # blank tile

    # moves: UP (-1, 0), DOWN (1, 0), LEFT (0, -1), RIGHT (0, 1)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n = len(state)

    for dr, dc in moves:
        r_prime, c_prime = r + dr, c + dc
        if 0 <= r_prime < n and 0 <= c_prime < n:
            new_state = [row[:] for row in state]  # deep copy of the state
            new_state[r][c], new_state[r_prime][c_prime] = (
                new_state[r_prime][c_prime],
                new_state[r][c],
            )
            neighbors.append(new_state)

    return neighbors


def heuristic_hamming(state, goal_state):
    misplaced = 0
    n = len(state)
    for r in range(n):
        for c in range(n):
            tile = state[r][c]
            if tile != 0 and tile != goal_state[r][c]:
                misplaced += 1
    return misplaced


def heuristic_manhattan(state, goal_state):
    manhattan_dist = 0
    n = len(state)
    for r in range(n):
        for c in range(n):
            tile = state[r][c]
            if tile != 0:
                target_r, target_c = get_position(goal_state, tile)
                manhattan_dist += abs(r - target_r) + abs(c - target_c)
    return manhattan_dist


def heuristic_manhattan_linear_conflict(state, goal_state):
    manhattan_dist = heuristic_manhattan(state, goal_state)
    linear_conflict = 0
    n = len(state)

    # check rows for linear conflicts
    for r in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                tile_i = state[r][i]
                tile_j = state[r][j]
                if tile_i != 0 and tile_j != 0:
                    target_r_i, target_c_i = get_position(goal_state, tile_i)
                    target_r_j, target_c_j = get_position(goal_state, tile_j)
                    if target_r_i == r and target_r_j == r and target_c_i > target_c_j:
                        linear_conflict += 2

    # check columns for linear conflicts
    for c in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                tile_i = state[i][c]
                tile_j = state[j][c]
                if tile_i != 0 and tile_j != 0:
                    target_r_i, target_c_i = get_position(goal_state, tile_i)
                    target_r_j, target_c_j = get_position(goal_state, tile_j)
                    if target_c_i == c and target_c_j == c and target_r_i > target_r_j:
                        linear_conflict += 2

    return manhattan_dist + linear_conflict


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]


def a_star(start_state, goal_state, heuristic):
    open_list = []  # priority queue of nodes ordered by f(n)
    closed_set = set()  # explored states

    start_node = Node(start_state, g=0, h=heuristic(start_state, goal_state))
    heapq.heappush(open_list, start_node)

    nodes_evaluated = 0

    while open_list:
        current = heapq.heappop(open_list)
        nodes_evaluated += 1

        if current.state == goal_state:
            return reconstruct_path(current), nodes_evaluated

        current_tuple = state_to_tuple(current.state)
        if current_tuple in closed_set:
            continue
        closed_set.add(current_tuple)

        for neighbor_state in generate_neighbors(current.state):
            neighbor_tuple = state_to_tuple(neighbor_state)

            if neighbor_tuple in closed_set:
                continue

            tentative_g = current.g + 1

            h = heuristic(neighbor_state, goal_state)
            neighbor_node = Node(neighbor_state, g=tentative_g, h=h, parent=current)
            heapq.heappush(open_list, neighbor_node)

    return None, nodes_evaluated  # no solution found


def main():
    start_state = [
        [12, 1, 2, 15],
        [11, 6, 5, 8],
        [7, 10, 9, 4],
        [0, 13, 14, 3],
    ]
    goal_state = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0],
    ]

    path, nodes_evaluated = a_star(
        start_state,
        goal_state,
        heuristic_manhattan_linear_conflict,
    )

    if path:
        print(
            f"Solution found in {len(path) - 1} moves, ",
            f"evaluated {nodes_evaluated} nodes.",
        )
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
