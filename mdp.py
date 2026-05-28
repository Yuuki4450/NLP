import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from collections import deque

GRID_WIDTH = 50
GRID_HEIGHT = 50

EMPTY = 0
OBSTACLE = 1

RED = 2
ORANGE = 3
YELLOW = 4
GREEN = 5
CYAN = 6
BLUE = 7
PURPLE = 8

PATH = 9

STEP_REWARD = -1
GOAL_REWARD = 100
gamma = 0.9
theta = 1e-6

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

points = {
    "RED": (44, 44),
    "ORANGE": (6, 42),
    "YELLOW": (42, 6),
    "GREEN": (26, 25),
    "CYAN": (10, 10),
    "BLUE": (3, 4),
    "PURPLE": (40, 20)
}

point_values = {
    "RED": RED,
    "ORANGE": ORANGE,
    "YELLOW": YELLOW,
    "GREEN": GREEN,
    "CYAN": CYAN,
    "BLUE": BLUE,
    "PURPLE": PURPLE
}

# =========================
# NLP 模块输出示例
# =========================

navigation_data = {
    "intent": "navigation",
    "start": "blue",
    "waypoints": ["orange"],
    "avoid": ["purple"],
    "end": "red",
    "is_complete": True,
    "missing_slots": []
}

cmap = mcolors.ListedColormap([
    'white',
    'black',
    'red',
    'orange',
    'yellow',
    'green',
    'cyan',
    'blue',
    'purple',
    'magenta'
])

bounds = np.arange(-0.5, 10.5, 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

def bfs(grid, start, goal):
    queue = deque([start])
    visited = set()

    while queue:
        r, c = queue.popleft()

        if (r, c) == goal:
            return True

        if (r, c) in visited:
            continue

        visited.add((r, c))

        for dr, dc in ACTIONS:
            nr = r + dr
            nc = c + dc

            if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                if grid[nr, nc] != OBSTACLE and (nr, nc) not in visited:
                    queue.append((nr, nc))

    return False

def all_reachable(grid):
    names = list(points.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1 = points[names[i]]
            p2 = points[names[j]]

            if not bfs(grid, p1, p2):
                return False

    return True

def carve_path(grid, p1, p2):
    r1, c1 = p1
    r2, c2 = p2

    r = r1
    c = c1

    while (r, c) != (r2, c2):
        grid[r, c] = EMPTY
        moves = []

        if r < r2:
            moves.append((1, 0))
        if r > r2:
            moves.append((-1, 0))
        if c < c2:
            moves.append((0, 1))
        if c > c2:
            moves.append((0, -1))

        if random.random() < 0.45:
            random_moves = [(1,0),(-1,0),(0,1),(0,-1)]
            random.shuffle(random_moves)

            for move in random_moves:
                nr = r + move[0]
                nc = c + move[1]
                if 1 <= nr < GRID_HEIGHT - 1 and 1 <= nc < GRID_WIDTH - 1:
                    moves.append(move)

        dr, dc = random.choice(moves)
        r += dr
        c += dc

        for _ in range(random.randint(1, 3)):
            rr = r + random.randint(-2, 2)
            cc = c + random.randint(-2, 2)
            if 0 <= rr < GRID_HEIGHT and 0 <= cc < GRID_WIDTH:
                grid[rr, cc] = EMPTY

    grid[r2, c2] = EMPTY

def generate_map():
    while True:
        grid = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)

        for _ in range(350):
            r = random.randint(1, GRID_HEIGHT - 8)
            c = random.randint(1, GRID_WIDTH - 8)
            h = random.randint(2, 9)
            w = random.randint(2, 9)
            grid[r:r+h, c:c+w] = EMPTY

        names = list(points.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                carve_path(grid, points[names[i]], points[names[j]])

        for _ in range(300):
            r = random.randint(1, GRID_HEIGHT - 5)
            c = random.randint(1, GRID_WIDTH - 5)
            h = random.randint(1, 4)
            w = random.randint(1, 4)
            if random.random() < 0.55:
                grid[r:r+h, c:c+w] = OBSTACLE

        for pos in points.values():
            r, c = pos
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        grid[nr, nc] = EMPTY

        for name, pos in points.items():
            grid[pos] = point_values[name]

        if all_reachable(grid):
            return grid

def solve_mdp(grid, goal_pos, avoid_positions=None):
    if avoid_positions is None:
        avoid_positions = []
    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    policy = np.full((GRID_HEIGHT, GRID_WIDTH), -1)

    value_function[goal_pos] = GOAL_REWARD
    iterations = 0

    while True:
        delta = 0

        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if grid[r, c] == OBSTACLE:
                    continue

                if (r, c) in avoid_positions:
                    continue

                if (r, c) == goal_pos:
                    continue

                old_v = value_function[r, c]
                q_values = []

                for action_idx, (dr, dc) in enumerate(ACTIONS):
                    nr = r + dr
                    nc = c + dc

                    if not (0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH):
                        q_values.append(-np.inf)
                        continue

                    if grid[nr, nc] == OBSTACLE:
                        q_values.append(-np.inf)
                        continue

                    if (nr, nc) in avoid_positions:
                        q_values.append(-99999)
                        continue

                    q = STEP_REWARD + gamma * value_function[nr, nc]
                    q_values.append(q)

                best_q = max(q_values)
                value_function[r, c] = best_q
                policy[r, c] = np.argmax(q_values)

                # 【修复核心 1】：隔离无处可去的死格子，规避 -inf - (-inf) = NaN 导致的异常和死循环
                if best_q != -np.inf:
                    delta = max(delta, abs(old_v - best_q))

        iterations += 1
        if delta < theta:
            break

    return value_function, policy, iterations

def extract_path(
    grid,
    policy,
    start_pos,
    goal_pos,
    avoid_positions=None
):
    if avoid_positions is None:
        avoid_positions = []
    visited = set()
    current = start_pos
    path = [current]
    max_steps = GRID_WIDTH * GRID_HEIGHT * 3

    for _ in range(max_steps):
        if current in visited:
            return None
        visited.add(current)

        if current == goal_pos:
            return path

        r, c = current
        action = policy[r, c]

        if action == -1:
            return None

        dr, dc = ACTIONS[action]
        nr = r + dr
        nc = c + dc

        if not (0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH):
            return None

        if grid[nr, nc] == OBSTACLE:
            return None

        if (nr, nc) in avoid_positions:
            return None
        current = (nr, nc)
        path.append(current)

    return None

def visualize(
    grid,
    value_function,
    policy,
    path,
    title="MDP Navigation"
):
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
    plt.title("Original 50x50 Map")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, GRID_WIDTH, 5))
    ax.set_yticks(np.arange(0, GRID_HEIGHT, 5))
    ax.set_xticks(np.arange(-0.5, GRID_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_HEIGHT, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.15, alpha=0.5)
    ax.grid(which='major', visible=False)
    ax.tick_params(axis='both', labelsize=8)
    plt.show()

    path_grid = np.copy(grid)
    if path:
        for r, c in path:
            if path_grid[r, c] == EMPTY:
                path_grid[r, c] = PATH

    plt.figure(figsize=(12, 12))
    plt.imshow(path_grid, cmap=cmap, norm=norm, origin='lower')
    plt.title(title)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, GRID_WIDTH, 5))
    ax.set_yticks(np.arange(0, GRID_HEIGHT, 5))
    ax.set_xticks(np.arange(-0.5, GRID_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_HEIGHT, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.15, alpha=0.5)
    ax.grid(which='major', visible=False)
    ax.tick_params(axis='both', labelsize=8)

    if path:
        plt.plot(
            [p[1] for p in path],
            [p[0] for p in path],
            color='white',
            linewidth=2
        )
    plt.show()

    plt.figure(figsize=(12, 12))
    heatmap = np.copy(value_function)
    heatmap[grid == OBSTACLE] = np.nan
    plt.imshow(heatmap, cmap='plasma', origin='lower')
    plt.title("Value Heatmap")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, GRID_WIDTH, 5))
    ax.set_yticks(np.arange(0, GRID_HEIGHT, 5))
    ax.set_xticks(np.arange(-0.5, GRID_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_HEIGHT, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.15, alpha=0.5)
    ax.grid(which='major', visible=False)
    ax.tick_params(axis='both', labelsize=8)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
    plt.title("Policy Visualization")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, GRID_WIDTH, 5))
    ax.set_yticks(np.arange(0, GRID_HEIGHT, 5))
    ax.set_xticks(np.arange(-0.5, GRID_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_HEIGHT, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.15, alpha=0.5)
    ax.grid(which='major', visible=False)
    ax.tick_params(axis='both', labelsize=8)

    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if grid[r, c] != OBSTACLE:
                action = policy[r, c]
                if action != -1:
                    dr, dc = ACTIONS[action]
                    plt.arrow(
                        c,
                        r,
                        dc * 0.35,
                        dr * 0.35,
                        head_width=0.15,
                        head_length=0.15,
                        fc='white',
                        ec='white'
                    )
    plt.show()

def run_navigation(nav_data):

    start = nav_data["start"].upper()
    end = nav_data["end"].upper()

    waypoints = [w.upper() for w in nav_data.get("waypoints", [])]
    avoid = [a.upper() for a in nav_data.get("avoid", [])]

    route = [start] + waypoints + [end]

    avoid_positions = [points[a] for a in avoid]

    grid = generate_map()

    full_path = []

    last_value_function = None
    last_policy = None

    total_iterations = 0

    for i in range(len(route) - 1):

        current_start = route[i]
        current_goal = route[i + 1]

        start_pos = points[current_start]
        goal_pos = points[current_goal]

        value_function, policy, iterations = solve_mdp(
            grid,
            goal_pos,
            avoid_positions
        )

        path = extract_path(
            grid,
            policy,
            start_pos,
            goal_pos,
            avoid_positions
        )

        if path is None:
            print(f"No Path Found: {current_start} -> {current_goal}")
            return

        if len(full_path) > 0:
            path = path[1:]

        full_path.extend(path)

        total_iterations += iterations

        last_value_function = value_function
        last_policy = policy

    print("Navigation Success")
    print("Route:", route)
    print("Avoid:", avoid)
    print("Path Length:", len(full_path) - 1)
    print("Total Iterations:", total_iterations)

    visualize(
        grid,
        last_value_function,
        last_policy,
        full_path,
        title=f"{' -> '.join(route)}"
    )

if __name__ == "__main__":

    navigation_data = {
        "intent": "navigation",
        "start": "blue",
        "waypoints": ["orange"],
        "avoid": ["purple"],
        "end": "red",
        "is_complete": True,
        "missing_slots": []
    }

    run_navigation(navigation_data)