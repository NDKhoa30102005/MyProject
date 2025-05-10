import pygame, sys
import random
from collections import deque
import heapq
import time
import math

pygame.init()

execution_time = 0

WIDTH, HEIGHT = 1100, 600
x_state = 150
y_state = 50
x_goal = 600
y_goal = 50

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Puzzle Solver")

BLUE_BG = (64, 90, 136)
TEXT_COLOR = (0, 0, 0)
TILE_COLOR = (242, 241, 220)
HIGHLIGHT = (255, 255, 240)
SHADOW = (158, 158, 158)

TILE_SIZE = 250 // 3
FONT = pygame.font.Font(None, 40)
FONT_ACTION = pygame.font.Font(None, 30)
STATE = [0] * 9  # Trạng thái ban đầu toàn ô trống (0)
GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Mục tiêu

selected_index = None  # Lưu ô đang được chọn để nhập số
pressed_button = None  # Nút đang được nhấn

BUTTONS = [
    ("BFS", 50, 520),
    ("DFS", 180, 520),
    ("A*", 310, 520),
    ("Greedy", 440, 520),
    ("UCS", 570, 520),
    ("SHC", 570, 460),
    ("IDDFS", 700, 520),
    ("IDA*", 830, 520),
    ("HC", 700, 400),
    ("SAHC", 700, 460),
    ("Genetic", 570, 400),
    ("Random", 830, 400),
    ("Reset", 830, 460),
    ("Beam", 960, 520),
    ("SA", 960, 460),
    ("and-or", 960, 400),
    ("Belief", 960, 340),
]

def is_solvable(state):
    """Kiểm tra trạng thái có thể giải được không"""
    inversions = 0
    for i in range(8):
        for j in range(i + 1, 9):
            if state[i] and state[j] and state[i] > state[j]:
                inversions += 1
    return inversions % 2 == 0

class SearchNode:
    """Định nghĩa một nút trong cây tìm kiếm"""
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        """So sánh dựa trên chi phí"""
        return self.cost < other.cost

def extract_path(node):
    """Trích xuất đường đi từ nút đích"""
    path = []
    action = []
    cost = node.cost
    while node.parent:
        action.append(node.action)
        path.append(node.state)
        node = node.parent
    path.reverse()
    action.reverse()
    return path,action, cost

def move(state, action):
    """Di chuyển ô trống"""
    new_state = state[:]
    i = new_state.index(0)  # Tìm vị trí ô trống

    if action == "up" and i >= 3:
        new_state[i], new_state[i - 3] = new_state[i - 3], new_state[i]
    elif action == "down" and i < 6:
        new_state[i], new_state[i + 3] = new_state[i + 3], new_state[i]
    elif action == "left" and i % 3 != 0:
        new_state[i], new_state[i - 1] = new_state[i - 1], new_state[i]
    elif action == "right" and i % 3 != 2:
        new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
    else:
        return None

    return new_state

def bfs_solve(initial_state, goal_state):
    frontier = deque([SearchNode(initial_state, cost=0)])
    explored = set()

    while frontier:
        node = frontier.popleft()
        if node.state == goal_state:
            return extract_path(node)

        explored.add(tuple(node.state))
        
        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state and tuple(new_state) not in explored:
                frontier.append(SearchNode(new_state, node, action, node.cost + 1))

    return None,None, 0

def dfs_solve(initial_state, goal_state, depth=100):
    frontier = [SearchNode(initial_state, cost=0)]
    explored = set()

    while frontier:
        node = frontier.pop()
        if node.state == goal_state:
            return extract_path(node)

        if node.cost < depth:
            explored.add(tuple(node.state))
            for action in ["up", "down", "left", "right"]:
                new_state = move(node.state, action)
                if new_state and tuple(new_state) not in explored:
                    frontier.append(SearchNode(new_state, node, action, node.cost + 1))

    return None,None, 0

def iterative_deepening_dfs_solve(initial_state, goal_state):
    for depth in range(1, 100):
        solution, action, cost = dfs_solve(initial_state, goal_state, depth)
        if solution:
            return solution, action, cost

    return None,None, 0
def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(9):
        if state[i] != 0:
            x1, y1 = i % 3, i // 3
            x2, y2 = goal_state.index(state[i]) % 3, goal_state.index(state[i]) // 3
            distance += abs(x1 - x2) + abs(y1 - y2)
    return distance
     
def greedy_best_first_search(initial_state, goal_state):
    frontier = [SearchNode(initial_state, cost=0)]
    explored = set()

    while frontier:
        frontier.sort(key=lambda node: manhattan_distance(node.state, goal_state))
        node = frontier.pop(0)
        if node.state == goal_state:
            return extract_path(node)

        explored.add(tuple(node.state))
        
        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state and tuple(new_state) not in explored:
                frontier.append(SearchNode(new_state, node, action, node.cost + 1))

    return None,None, 0

def a_star_search(initial_state, goal_state):
    frontier = [SearchNode(initial_state, cost=0)]
    explored = set()

    while frontier:
        frontier.sort(key=lambda node: node.cost + manhattan_distance(node.state, goal_state))
        node = frontier.pop(0)
        if node.state == goal_state:
            return extract_path(node)

        explored.add(tuple(node.state))
        
        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state and tuple(new_state) not in explored:
                frontier.append(SearchNode(new_state, node, action, node.cost + 1))

    return None,None, 0

def uniform_cost_search(initial_state, goal_state):
    frontier = []  # Priority queue
    heapq.heappush(frontier, (0, SearchNode(initial_state, cost=0)))
    explored = set()
    cost_so_far = {tuple(initial_state): 0}  # Track lowest cost to reach each state
    
    while frontier:
        cost, node = heapq.heappop(frontier)
        
        if node.state == goal_state:
            return extract_path(node)
        
        explored.add(tuple(node.state))
        
        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state:
                new_state_tuple = tuple(new_state)
                new_cost = node.cost + 1
                
                if new_state_tuple not in explored and (new_state_tuple not in cost_so_far or new_cost < cost_so_far[new_state_tuple]):
                    cost_so_far[new_state_tuple] = new_cost
                    heapq.heappush(frontier, (new_cost, SearchNode(new_state, node, action, new_cost)))
    
    return None,None, 0

def ida_star_search(STATE, GOAL):
    def search(node, g, bound):
        f = g + manhattan_distance(node.state, GOAL)
        if f > bound:
            return f, None  # Trả về ngưỡng mới để tăng dần

        if node.state == GOAL:
            return -1, node  # Khi tìm thấy lời giải, trả về nút cuối cùng

        min_cost = float("inf")
        best_solution = None  # Lưu trạng thái của nút lời giải nếu tìm thấy

        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state and new_state != node.state:
                new_node = SearchNode(new_state, node, action, g + 1)  # Dùng g + 1 thay vì node.cost + 1
                cost, result_node = search(new_node, g + 1, bound)

                if cost == -1:
                    return -1, result_node  # Trả về nút chứa trạng thái đích

                if cost < min_cost:
                    min_cost = cost

        return min_cost, best_solution

    bound = manhattan_distance(STATE, GOAL)
    node = SearchNode(STATE)

    while True:
        cost, result_node = search(node, 0, bound)
        if cost == -1:
            return extract_path(result_node)  # Dùng nút kết thúc để lấy đường đi

        if cost == float("inf"):
            return None, None, 0  # Không tìm thấy đường đi

        bound = cost  # Cập nhật ngưỡng mới nếu chưa tìm thấy

def hill_climbing(STATE, GOAL):
    def get_heuristic(state):
        return manhattan_distance(state, GOAL)

    current = SearchNode(STATE)
    
    for _ in range(100):  # Thử lại tối đa 100 lần nếu bị mắc kẹt
        if current.state == GOAL:
            return extract_path(current)
        
        neighbors = []
        for action in ["up", "down", "left", "right"]:
            new_state = move(current.state, action)
            if new_state:
                neighbors.append(SearchNode(new_state, current, action, current.cost + 1))

        if not neighbors:
            break  

        neighbors.sort(key=lambda node: get_heuristic(node.state))
        best_neighbor = neighbors[0]

        if get_heuristic(best_neighbor.state) >= get_heuristic(current.state):  
            current = SearchNode(random.sample(STATE, len(STATE)))  # Restart nếu mắc kẹt
        else:
            current = best_neighbor  

    return None, None, 0  # Trả về None nếu không tìm được

def sahc(STATE, GOAL):
    def get_heuristic(state):
        return manhattan_distance(state, GOAL)

    current = SearchNode(STATE)
    
    for _ in range(100):  # Giới hạn số lần mắc kẹt
        if current.state == GOAL:
            return extract_path(current)

        neighbors = []
        for action in ["up", "down", "left", "right"]:
            new_state = move(current.state, action)
            if new_state:
                neighbors.append(SearchNode(new_state, current, action, current.cost + 1))

        if not neighbors:
            break  

        best_neighbor = min(neighbors, key=lambda node: get_heuristic(node.state))

        if get_heuristic(best_neighbor.state) >= get_heuristic(current.state):  
            current = SearchNode(random.sample(STATE, len(STATE)))  # Restart nếu mắc kẹt
        else:
            current = best_neighbor  

    return None, None, 0  # Không tìm thấy lời giải

def shc(STATE, GOAL):
    def get_heuristic(state):
        return manhattan_distance(state, GOAL)

    current = SearchNode(STATE)
    
    for _ in range(100):  # Giới hạn số lần mắc kẹt
        if current.state == GOAL:
            return extract_path(current)

        neighbors = []
        for action in ["up", "down", "left", "right"]:
            new_state = move(current.state, action)
            if new_state:
                neighbors.append(SearchNode(new_state, current, action, current.cost + 1))

        if not neighbors:
            break  

        best_neighbor = min(neighbors, key=lambda node: get_heuristic(node.state))

        if get_heuristic(best_neighbor.state) >= get_heuristic(current.state):  
            current = SearchNode(random.sample(STATE, len(STATE)))  # Restart nếu mắc kẹt
        else:
            current = best_neighbor  

    return None, None, 0  # Không tìm thấy lời giải

def genetic_algorithm(STATE, GOAL, population_size=100, generations=1000, mutation_rate=0.1):
    def get_heuristic(state):
        return manhattan_distance(state, GOAL)
    
    def crossover(parent1, parent2):
        crossover_point = random.randint(1, 7)
        child = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
        return child
    
    def mutate(state):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(9), 2)
            state[idx1], state[idx2] = state[idx2], state[idx1]
        return state
    
    population = [random.sample(STATE, len(STATE)) for _ in range(population_size)]
    
    for _ in range(generations):
        population.sort(key=get_heuristic)
        if get_heuristic(population[0]) == 0:
            return extract_path(SearchNode(population[0]))
        
        new_population = population[:10]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:50], 2)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)
        
        population = new_population
    
    return None  # Không tìm thấy lời giải

def beam_search(initial_state, goal_state, beam_width=3):
    frontier = [SearchNode(initial_state, cost=0)]
    explored = set()

    while frontier:
        frontier.sort(key=lambda node: manhattan_distance(node.state, goal_state))
        if len(frontier) > beam_width:
            frontier = frontier[:beam_width]

        node = frontier.pop(0)
        if node.state == goal_state:
            return extract_path(node)

        explored.add(tuple(node.state))

        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state and tuple(new_state) not in explored:
                frontier.append(SearchNode(new_state, node, action, node.cost + 1))

    return None,None, 0

def simulated_annealing(STATE, GOAL, initial_temp=1000, cooling_rate=0.99):
    def get_heuristic(state):
        return manhattan_distance(state, GOAL)
    current = SearchNode(STATE)
    temp = initial_temp
    while temp > 0.001:
        if current.state == GOAL:
            return extract_path(current)
        neighbors = []
        for action in ["up", "down", "left", "right"]:
            new_state = move(current.state, action)
            if new_state:
                neighbors.append(SearchNode(new_state, current, action, current.cost + 1))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        delta_e = get_heuristic(next_node.state) - get_heuristic(current.state)
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            current = next_node  
        temp *= cooling_rate  
    return None, None, 0  # Không tìm thấy lời giải

def and_or_search(initial_state, goal_state):
    """AND-OR search for the 8-puzzle problem"""
    def recursive_search(node, explored):
        """Recursively search for a solution"""
        if node.state == goal_state:
            return [node.state], [], node.cost  # Return path, actions, and cost
        
        state_tuple = tuple(node.state)
        if state_tuple in explored:
            return None, None, float('inf')  # Avoid cycles
        
        explored.add(state_tuple)
        best_path = None
        best_actions = None
        best_cost = float('inf')
        
        # Explore all possible actions (OR node)
        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state:
                # Create a new node for the child state
                child_node = SearchNode(new_state, node, action, node.cost + 1)
                # Recursively search the child state
                child_path, child_actions, child_cost = recursive_search(child_node, explored.copy())
                
                # If a valid path is found and its cost is lower, update the best solution
                if child_path is not None and child_cost < best_cost:
                    best_path = [node.state] + child_path
                    best_actions = [action] + child_actions
                    best_cost = child_cost
        
        explored.remove(state_tuple)  # Backtrack to allow other paths to explore this state
        if best_path is None:
            return None, None, float('inf')
        return best_path, best_actions, best_cost

    # Initialize the search with the initial state
    initial_node = SearchNode(initial_state, cost=0)
    explored = set()
    path, actions, cost = recursive_search(initial_node, explored)
    
    if path is None:
        return None, None, 0  # No solution found
    return path, actions, cost

def belief_search(initial_state, goal_state):
    """A probabilistic variant of A* search simulating belief state uncertainty"""
    def probabilistic_heuristic(state, goal_state):
        # Add randomness to Manhattan distance to simulate uncertainty
        base_heuristic = manhattan_distance(state, goal_state)
        noise = random.uniform(-1, 1)  # Small random noise
        return base_heuristic + noise

    frontier = [(0, SearchNode(initial_state, cost=0))]
    explored = set()

    while frontier:
        frontier.sort(key=lambda x: x[0] + probabilistic_heuristic(x[1].state, goal_state))
        _, node = frontier.pop(0)
        if node.state == goal_state:
            return extract_path(node)

        explored.add(tuple(node.state))

        for action in ["up", "down", "left", "right"]:
            new_state = move(node.state, action)
            if new_state and tuple(new_state) not in explored:
                new_cost = node.cost + 1
                frontier.append((new_cost, SearchNode(new_state, node, action, new_cost)))

    return None, None, 0

def randomize_state():
    """Tạo trạng thái ngẫu nhiên không cần kiểm tra tính khả thi"""
    global STATE
    STATE = GOAL[:]  # Sao chép trạng thái GOAL
    random.shuffle(STATE)  # Xáo trộn ngẫu nhiên

def draw_button(x, y, text, is_pressed=False):
    """Vẽ nút có hiệu ứng nhấn"""
    width, height = 120, 50
    rect = pygame.Rect(x, y, width, height)

    bg_color = (230, 180, 100) if is_pressed else TILE_COLOR  
    pygame.draw.rect(SCREEN, bg_color, rect)  
    pygame.draw.rect(SCREEN, SHADOW, rect, 3)  

    pygame.draw.polygon(SCREEN, HIGHLIGHT, [(x, y), (x + width, y), (x + width - 3, y + 3), (x + 3, y + 3)])
    pygame.draw.polygon(SCREEN, HIGHLIGHT, [(x, y), (x, y + height), (x + 3, y + height - 3), (x + 3, y + 3)])

    pygame.draw.polygon(SCREEN, SHADOW, [(x + width, y), (x + width, y + height), (x + width - 3, y + height - 3), (x + width - 3, y + 3)])
    pygame.draw.polygon(SCREEN, SHADOW, [(x, y + height), (x + width, y + height), (x + width - 3, y + height - 3), (x + 3, y + height - 3)])

    text_surf = FONT.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    SCREEN.blit(text_surf, text_rect)

    return rect  

def get_tile_index(x, y, start_x, start_y):
    """Xác định chỉ số ô trong mảng dựa trên vị trí chuột"""
    if start_x <= x < start_x + 3 * TILE_SIZE and start_y <= y < start_y + 3 * TILE_SIZE:
        col = (x - start_x) // TILE_SIZE
        row = (y - start_y) // TILE_SIZE
        return row * 3 + col
    return None

def draw_grid(start_x, start_y, grid_data, selected_idx):
    """Vẽ lưới 3x3 tại vị trí chỉ định"""
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            num = grid_data[index]
            x = start_x + col * TILE_SIZE
            y = start_y + row * TILE_SIZE
            draw_tile(x, y, num, index == selected_idx)

def draw_tile(x, y, num, is_selected):
    """Vẽ một ô trong bảng"""
    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
    color = (255, 220, 150) if is_selected else TILE_COLOR  
    pygame.draw.rect(SCREEN, color, rect)

    pygame.draw.polygon(SCREEN, HIGHLIGHT, [(x, y), (x + TILE_SIZE, y), (x + TILE_SIZE - 5, y + 5), (x + 5, y + 5)])
    pygame.draw.polygon(SCREEN, HIGHLIGHT, [(x, y), (x, y + TILE_SIZE), (x + 5, y + TILE_SIZE - 5), (x + 5, y + 5)])
    pygame.draw.polygon(SCREEN, SHADOW, [(x + TILE_SIZE, y), (x + TILE_SIZE, y + TILE_SIZE),
                                         (x + TILE_SIZE - 5, y + TILE_SIZE - 5), (x + TILE_SIZE - 5, y + 5)])
    pygame.draw.polygon(SCREEN, SHADOW, [(x, y + TILE_SIZE), (x + TILE_SIZE, y + TILE_SIZE),
                                         (x + TILE_SIZE - 5, y + TILE_SIZE - 5), (x + 5, y + TILE_SIZE - 5)])

    if num != 0:
        text = FONT.render(str(num), True, TEXT_COLOR)
        text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
        SCREEN.blit(text, text_rect)

button_rects = {text: draw_button(x, y, text) for text, x, y in BUTTONS}
click = None
solution_path = None
action = None
cost = 0

while True:
    SCREEN.fill(BLUE_BG)
    mouse_x, mouse_y = pygame.mouse.get_pos()

    text_state = FONT.render("State", True, TEXT_COLOR)
    SCREEN.blit(text_state, (x_state + 90, y_state - 30))
    text_goal = FONT.render("Goal", True, TEXT_COLOR)
    SCREEN.blit(text_goal, (x_goal + 90, y_goal - 30))

    draw_grid(x_state, y_state, STATE, selected_index)
    draw_grid(x_goal, y_goal, GOAL, None)

    button_rects = {text: draw_button(x, y, text, pressed_button == text) for text, x, y in BUTTONS}

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            for text, rect in button_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    pressed_button = text
                    break
            else:
                pressed_button = None  
                selected_index = get_tile_index(mouse_x, mouse_y, x_state, y_state)

        elif event.type == pygame.MOUSEBUTTONUP:
            if pressed_button and button_rects[pressed_button].collidepoint(mouse_x, mouse_y):
                print(f"{pressed_button} button clicked!")
                click = pressed_button

        elif event.type == pygame.KEYDOWN and selected_index is not None:
            if pygame.K_1 <= event.key <= pygame.K_8:
                num = event.key - pygame.K_0
                STATE[selected_index] = num
                selected_index = None
            elif event.key == pygame.K_BACKSPACE:
                STATE[selected_index] = 0
                selected_index = None
    if click:
        start_time = time.time()
        
        if is_solvable(STATE):
            if click == "BFS":
                solution_path,action, cost = bfs_solve(STATE, GOAL)
            elif click == "DFS":
                solution_path,action, cost = dfs_solve(STATE, GOAL)
            elif click == "IDDFS":
                solution_path,action, cost = iterative_deepening_dfs_solve(STATE, GOAL)
            elif click == "UCS":
                solution_path,action, cost = uniform_cost_search(STATE, GOAL)
            elif click == "Greedy":
                solution_path,action, cost = greedy_best_first_search(STATE, GOAL)
            elif click == "A*":
                solution_path,action, cost = a_star_search(STATE, GOAL)
            elif click == "Genetic":
                solution_path,action, cost = genetic_algorithm(STATE, GOAL)
            elif click == "IDA*":
                solution_path,action, cost = ida_star_search(STATE, GOAL)
            elif click == "Beam":
                solution_path,action, cost = beam_search(STATE, GOAL)
            elif click == "SA":
                solution_path,action, cost = simulated_annealing(STATE, GOAL)
            elif click == "HC":
                while True:
                    solution_path,action, cost = ida_star_search(STATE, GOAL)
                    if solution_path:
                        break
            elif click == "SAHC":
                while True:
                    solution_path,action, cost = sahc(STATE, GOAL)
                    if solution_path:
                        break
            elif click == "SHC":
                while True:
                    solution_path,action, cost = shc(STATE, GOAL)
                    if solution_path:
                        break
            elif click == "and-or":
                solution_path,action, cost = and_or_search(STATE, GOAL)
            elif click == "Belief":
                solution_path,action, cost = belief_search(STATE, GOAL)
            
        if click == "Random":
            randomize_state()
            solution_path = None
            action = None
            cost = 0
        elif click == "Reset":
            STATE = [0] * 9
            selected_index = None
            solution_path = None
            action = None
            cost = 0    
        execution_time = time.time() - start_time 
        click = None

    if is_solvable(STATE):
        text_solvable = FONT.render("Solvable", True, TEXT_COLOR)
    else:
        text_solvable = FONT.render("Unsolvable", True, TEXT_COLOR)
    SCREEN.blit(text_solvable, (830, 350))

    text_cost = FONT.render(f"Cost: {cost}", True, TEXT_COLOR)
    SCREEN.blit(text_cost, (70, 320))
    
    text_action = FONT_ACTION.render(f"Action:", True, TEXT_COLOR)
    SCREEN.blit(text_action, (70, 350))

    text_time = FONT.render(f"Time: {execution_time:.6f}", True, TEXT_COLOR)
    SCREEN.blit(text_time, (200, 320))

    y_position = 320
    LINE_SPACING = 30
    WORDS_PER_LINE = 11
    x_position = 150

    y_text = y_position + LINE_SPACING
    if action:
        for i in range(0, len(action), WORDS_PER_LINE):
            line_text = " ".join(action[i:i + WORDS_PER_LINE])
            text_action = FONT_ACTION.render(line_text, True, TEXT_COLOR)  
            SCREEN.blit(text_action, (x_position, y_text))
            y_text += LINE_SPACING


    if solution_path:
        for state in solution_path:
            draw_grid(x_state, y_state, state, None)
            STATE = state
            pygame.display.flip()
            pygame.time.delay(500)
        solution_path = None

    pygame.display.update()
