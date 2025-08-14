import pygame
from simulation import Simulation
from sand_tetromino import SandTetromino
import time
import numpy as np
from tetris_nn import TetrisNet
import torch
from collections import deque
import random
import os
import threading
from scipy.ndimage import label
import cProfile, pstats
from contextlib import nullcontext
import json
import math

LOSS_HISTORY_PATH = "loss_history.json"
loss_history = []
_loss_accum = []
_loss_accum_lock = threading.Lock()

if os.path.exists(LOSS_HISTORY_PATH):
    try:
        with open(LOSS_HISTORY_PATH, 'r') as f:
            loss_history = json.load(f)
        print(f"Loaded loss history from {LOSS_HISTORY_PATH}")
    except Exception as e:
        print(f"Failed to load loss history: {e}")
else:
    print("No saved loss history found, starting fresh.")

pygame.init()
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

TITLE_FONT = pygame.font.SysFont(None, 32, bold=True)
TERM_FONT  = pygame.font.SysFont(None, 22)
RULES_FONT = pygame.font.SysFont(None, 24, bold=True)
PAUSE_FONT = pygame.font.SysFont(None, 60)

THEME_COLOR = (40, 40, 40)
PLAYFIELD_BG_COLOR = (0, 0, 0)

WINDOW_HEIGHT = 810
WINDOW_WIDTH = 1440

CELL_SIZE = 3
PLAYFIELD_WIDTH = 492
PLAYFIELD_HEIGHT = 780
PLAYFIELD_X = 450
SIDEBAR_WIDTH = WINDOW_WIDTH - PLAYFIELD_X - PLAYFIELD_WIDTH
LEFT_SIDEBAR_WIDTH = PLAYFIELD_X

FPS = 60
WALL_THICKNESS = 0
GHOST_ROWS = 2
VISIBLE_WINDOW_Y = GHOST_ROWS * CELL_SIZE
ACTION_INTERVAL = 3
move_hold_counter = 0
last_reward_components = {}

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Subscribe to Pleth")
simulation = Simulation(PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT, CELL_SIZE)

ground_y = PLAYFIELD_HEIGHT - 1
box_size = 36
spawn_y = -GHOST_ROWS * box_size
cumulative_score = 0.0

BASE_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 200, 0),
    (255, 255, 0),
]

last_sand_clear_score = 0

def get_centered_spawn_x(tetromino_class, playfield_width, box_size, **kwargs):
    """Calculate centered spawn X position for a tetromino piece."""
    temp = tetromino_class(0, 0, box_size, **kwargs)
    min_bx = min(box.x for box in temp.boxes)
    max_bx = max(box.x + box.size for box in temp.boxes)
    piece_width = max_bx - min_bx
    spawn_x = (playfield_width - piece_width) // 2 - min_bx
    return spawn_x

active_color = random.choice(BASE_COLORS)
next_color = random.choice(BASE_COLORS)
spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, box_size, color=active_color)
tetrominoes = []
active_tetromino = SandTetromino(spawn_x, spawn_y, box_size, color=active_color)
tetrominoes.append(active_tetromino)
next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, box_size, color=next_color)
next_piece = SandTetromino(next_spawn_x, spawn_y, box_size, color=next_color)

move_left = False
move_right = False
speed_down = False
move_speed = 10

game_over = False
pause = False

prev_broken = False

score = 0
level = 1
pixels_cleared_total = 0
generation = 1
max_score = 0
start_time = time.time()
generation_start_time = start_time
exploration_actions = 0
exploitation_actions = 0

soft_drop_frame_counter = 0
nn_soft_drop_frame_counter = 0

def get_points_for_pixels(pixels):
    """Calculate score points based on number of pixels cleared with multipliers."""
    if pixels < 100:
        return pixels * 1
    elif pixels < 400:
        return pixels * 1.1
    elif pixels < 900:
        return pixels * 1.2
    elif pixels < 1600:
        return pixels * 1.3
    elif pixels < 2500:
        return pixels * 1.4
    elif pixels < 3600:
        return pixels * 1.5
    elif pixels < 4900:
        return pixels * 1.6
    elif pixels < 6400:
        return pixels * 1.8
    else:
        return pixels * 2

def get_level(score):
    """Determine current level based on score with 5000 points per level."""
    return max(1, int(score // 5000) + 1)

def get_gravity(level):
    """Calculate gravity speed based on level with exponential scaling."""
    return 2.5 * 1.1**(level / 5)

def build_color_index(grid):
    """Build a color index array from the grid for reward calculations."""
    rows, cols = grid.rows, grid.columns
    colors = np.zeros((rows, cols), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            p = grid.cells[r][c]
            if p is not None:
                col = getattr(p, 'base_color', getattr(p, 'color', None))
                if col is not None:
                    if col[0] > 200 and col[1] < 100 and col[2] < 100:
                        colors[r, c] = 1
                    elif col[2] > 100 and col[0] < 100 and col[1] < 200:
                        colors[r, c] = 2
                    elif col[0] < 100 and col[1] > 150 and col[2] < 100:
                        colors[r, c] = 3
                    elif col[0] > 200 and col[1] > 200 and col[2] < 100:
                        colors[r, c] = 4
    
    return colors

def prepare_nn_input(grid):
    """Convert grid to neural network input tensor with one-hot encoding."""
    colors = build_color_index(grid)
    one_hot = np.eye(5, dtype=np.float32)[colors][..., 1:]
    nn_arr = one_hot.transpose(2, 0, 1)[None]
    return torch.from_numpy(nn_arr)

_label_cache = {}
_cache_frame = 0

last_reward_components['score'] = 0.0
live=0

def get_column_heights(color_index):
    """Calculate height of sand in each column."""
    rows, cols = color_index.shape
    heights = np.zeros(cols, dtype=int)
    present = color_index > 0
    max_indices = np.argmax(present, axis=0)
    has_sand = np.any(present, axis=0)
    heights[has_sand] = rows - max_indices[has_sand]
    return heights

def calculate_strategic_height_penalty(color_index):
    """Calculate penalty based on board height with extreme penalties for dangerous heights."""
    rows, cols = color_index.shape
    height_penalty = 0.0
    
    for color in range(1, 5):
        color_mask = (color_index == color)
        if not np.any(color_mask):
            continue
            
        labeled, num_features = label(color_mask)
        
        for i in range(1, num_features + 1):
            component_mask = (labeled == i)
            rows_with_color, cols_with_color = np.where(component_mask)
            
            if len(rows_with_color) == 0:
                continue
                
            top_row = np.min(rows_with_color)
            bottom_row = np.max(rows_with_color)
            left_col = np.min(cols_with_color)
            right_col = np.max(cols_with_color)
            
            height = bottom_row - top_row + 1
            width = right_col - left_col + 1
            
            if height > width * 1.5 and height > 10:
                height_penalty -= height * 2
    
    heights = get_column_heights(color_index)
    max_height = np.max(heights) if len(heights) > 0 else 0
    height_ratio = max_height / rows
    
    if height_ratio > 0.7:
        excess_ratio = height_ratio - 0.7
        scaled_excess = excess_ratio / 0.3
        extreme_penalty = -1500 - (scaled_excess ** 2) * 18500
        height_penalty += extreme_penalty
    else:
        if height_ratio > 0.5:
            avg_height = np.mean(heights)
            height_penalty -= avg_height * 5
        
    return height_penalty

def calculate_reward(color_index, score_diff, game_over):
    """Calculate reward based on game state with focus on spanning potential and score."""
    global score, last_reward_components
    if game_over:
        return -5000.0
    
    spanning_potential = calculate_spanning_potential(color_index)
    height_penalty = calculate_strategic_height_penalty(color_index)
    
    fragmentation_penalty = 0
    for color in range(1, 5):
        color_mask = (color_index == color)
        if np.any(color_mask):
            labeled_array, num_features = label(color_mask)
            if num_features > 2:
                fragmentation_penalty -= (num_features - 2) * 100
    
    clear_board_reward = -np.sum(get_column_heights(color_index)) * 0.1
    
    score_tier = int(score // 10000)
    score_multiplier = 0.4 * (1.2 ** score_tier)
    score_reward = score * score_multiplier

    total_reward = (
        spanning_potential * 1.5 +
        height_penalty +
        fragmentation_penalty +
        clear_board_reward +
        score_reward
    )

    last_reward_components = {
        'Spanning': spanning_potential * 1.5,
        'Height Penalty': height_penalty,
        'Fragmentation': fragmentation_penalty,
        'Clear Board': clear_board_reward,
        'Score': score_reward
    }
    
    return total_reward

def calculate_spanning_potential(color_index):
    """Calculate reward for colors that span from left to right walls."""
    rows, cols = color_index.shape
    total_reward = 0.0
    
    for color in range(1, 5):
        color_mask = (color_index == color)
        if not np.any(color_mask):
            continue
            
        for row in range(rows):
            if not np.any(color_mask[row, :]):
                continue
                
            cols_with_color = np.where(color_mask[row, :])[0]
            if len(cols_with_color) == 0:
                continue
                
            leftmost = cols_with_color[0]
            rightmost = cols_with_color[-1]
            span_width = rightmost - leftmost + 1
            
            density = len(cols_with_color) / span_width
            
            if span_width >= 3:
                width_reward = (span_width / cols * 2) ** 3
                density_bonus = density * width_reward
                row_reward = width_reward + density_bonus
                
                edge_bonus = 0
                if leftmost <= 2:
                    edge_bonus += 0.25
                if rightmost >= cols - 3:
                    edge_bonus += 0.25
                if leftmost <= 2 and rightmost >= cols - 3:
                    edge_bonus += 1.0
                
                total_reward += row_reward + edge_bonus
    
    return total_reward

ACTIONS = ['left', 'right', 'rotate_right', 'rotate_left', 'drop', 'none', 'hold_left', 'hold_right', 'hold_drop']
N_ACTIONS = len(ACTIONS)

MODEL_PATH = "tetris_brain.pth"
STEPS_PATH = "steps_done.json"

rows, cols = simulation.grid.rows, simulation.grid.columns
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
net = TetrisNet(input_channels=4,
                height=rows,
                width=cols,
                n_actions=N_ACTIONS).to(device, dtype=torch.float32)
if device.type == "mps":
    try:
        test_tensor = torch.randn(1, 4, 32, 32, device=device)
        print("MPS device test successful")
        del test_tensor
        torch.mps.empty_cache()
    except Exception as e:
        print(f"MPS test failed: {e}, falling back to CPU")
        device = torch.device("cpu")

model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model weights from {MODEL_PATH}")
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model_loaded = True
    except RuntimeError as e:
        print(f"Model load failed due to architecture mismatch: {e}\nDeleting old model file and starting fresh.")
        os.remove(MODEL_PATH)
        model_loaded = False
if not model_loaded:
    print("No saved model found, starting fresh.")

steps_done = 0
generation = 1
cumulative_score = 0.0
max_score = 0
start_time = time.time()
if os.path.exists(STEPS_PATH):
    try:
        with open(STEPS_PATH, 'r') as f:
            data = json.load(f)
            steps_done = data.get('steps_done', 0)
            generation = data.get('generation', 1)
            cumulative_score = data.get('cumulative_score', 0.0)
            max_score = data.get('max_score', 0)
            start_time = data.get('start_time', time.time())
            print(f"Loaded stats from {STEPS_PATH}")
    except Exception as e:
        print(f"Failed to load stats from {STEPS_PATH}: {e}")
else:
    print("No saved stats found, starting fresh.")

print("M2 Apple Silicon optimizations applied!")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

net.eval()

REPLAY_SIZE = 15000
BATCH_SIZE = 64
GAMMA = 0.995
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 7500000
LEARNING_RATE = 1e-5

replay_buffer = deque(maxlen=REPLAY_SIZE)
epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

last_state = prepare_nn_input(simulation.grid)
last_score = score

train_frame_counter = 0
training_thread = None
training_lock = threading.Lock()

TERMINAL_BUFFER_SIZE = 8
terminal_lines = []
terminal_highlight = []

import atexit
training_active = False
def dqn_train_step():
    """Perform one DQN training step with experience replay."""
    if len(replay_buffer) < BATCH_SIZE:
        return
    effective_batch_size = min(16, len(replay_buffer))
    batch = random.sample(replay_buffer, effective_batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.cat([s.unsqueeze(0) if s.ndim == 3 else s for s in states])
    next_states = torch.cat([s.unsqueeze(0) if s.ndim == 3 else s for s in next_states])
    states = states.to(device, non_blocking=True, dtype=torch.float32)
    next_states = next_states.to(device, non_blocking=True, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device)
    optimizer.zero_grad()
    q_vals = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = net(next_states).max(1)[0]
    target_q = rewards + GAMMA * next_q * (~dones)
    loss = loss_fn(q_vals, target_q)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    try:
        loss_val = float(loss.item())
        if not math.isnan(loss_val) and not math.isinf(loss_val):
            with _loss_accum_lock:
                _loss_accum.append(loss_val)
                if len(_loss_accum) >= 40:
                    avg_loss = sum(_loss_accum) / len(_loss_accum)
                    loss_history.append(avg_loss)
                    _loss_accum.clear()
                    try:
                        with open(LOSS_HISTORY_PATH, 'w') as f:
                            json.dump(loss_history, f)
                        print(f"Saved loss_history (len={len(loss_history)}) to {LOSS_HISTORY_PATH}")
                    except Exception as e:
                        print(f"Failed to save loss_history: {e}")
    except Exception as e:
        print(f"Loss tracking error: {e}")
    if device.type == "mps":
        torch.mps.empty_cache()
    del states, next_states, actions, rewards, dones, q_vals, next_q, target_q, loss

def training_worker():
    """Background worker thread for continuous training."""
    global training_active
    while training_active:
        with training_lock:
            dqn_train_step()
        time.sleep(0.25)

def start_training_thread():
    """Start the background training thread."""
    global training_thread, training_active
    if training_thread is None or not training_thread.is_alive():
        training_active = True
        training_thread = threading.Thread(target=training_worker, daemon=True)
        training_thread.start()

def stop_training_thread():
    """Stop the background training thread."""
    global training_active, training_thread
    training_active = False
    if training_thread and training_thread.is_alive():
        training_thread.join(timeout=2.0)

start_training_thread()
atexit.register(stop_training_thread)

def add_terminal_line(text, highlight=False):
    """Add a line to the terminal display buffer."""
    global terminal_lines, terminal_highlight

last_action = 'none'
last_action_idx = ACTIONS.index('none')

AI_RULESET = [
    "1. Clear sand for points",
    "2. Keep the sand low",
    "3. Avoid game over",
    "4. Avoid disconnects/full covers",
    "5. Reward larger connections",
    "6. Try to make a color go from L to R",
    "7. Keep pieces low"
]

_text_cache = {}
_cache_size_limit = 100

def render_text_cached(text, font, color):
    """Render text with caching to improve performance."""
    global _text_cache
    
    cache_key = (text, id(font), color)
    if cache_key in _text_cache:
        return _text_cache[cache_key]
    
    surface = font.render(text, True, color)
    
    if len(_text_cache) >= _cache_size_limit:
        keys_to_remove = list(_text_cache.keys())[:20]
        for key in keys_to_remove:
            del _text_cache[key]
    
    _text_cache[cache_key] = surface
    return surface

last_reward = 0.0
cumulative_reward = 0.0
running = True

pause_overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
pause_overlay.fill((0, 0, 0, 120))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            try:
                with open(STEPS_PATH, 'w') as f:
                    json.dump({'steps_done': steps_done}, f)
                print(f"Saved steps_done: {steps_done} to {STEPS_PATH}")
            except Exception as e:
                print(f"Failed to save steps_done to {STEPS_PATH}: {e}")
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                move_left = True
            elif event.key == pygame.K_RIGHT:
                move_right = True
            elif event.key == pygame.K_DOWN:
                speed_down = True
            elif event.key == pygame.K_UP:
                if all((box.y // CELL_SIZE) >= GHOST_ROWS for box in active_tetromino.boxes):
                    active_tetromino.rotate('right', wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS, grid=simulation.grid)
            elif event.key == pygame.K_z:
                if all((box.y // CELL_SIZE) >= GHOST_ROWS for box in active_tetromino.boxes):
                    active_tetromino.rotate('left', wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS, grid=simulation.grid)
            elif event.key == pygame.K_p:
                pause = not pause
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                move_left = False
            if event.key == pygame.K_RIGHT:
                move_right = False
            if event.key == pygame.K_DOWN:
                speed_down = False

    if pause:
        window.fill(THEME_COLOR)
        simulation.draw(window.subsurface((PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT)))
        for tetromino in tetrominoes:
            tetromino.draw(window.subsurface((PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT)))
        window.blit(pause_overlay, (0, 0))
        text = render_text_cached("PAUSED", PAUSE_FONT, (220, 220, 220))
        window.blit(text, ((WINDOW_WIDTH - text.get_width()) // 2, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        clock.tick(FPS)
        continue

    if game_over:
        total_actions = exploration_actions + exploitation_actions
        explore_ratio = (exploration_actions / total_actions * 100) if total_actions > 0 else 0
        exploit_ratio = (exploitation_actions / total_actions * 100) if total_actions > 0 else 0
        decay_progress = min(1.0, steps_done / EPSILON_DECAY) * 100
        
        print(f"\n{'='*60}")
        print(f"GAME OVER! Generation {generation}")
        print(f"Score: {int(score)} | Pixels cleared: {pixels_cleared_total}")
        print(f"Epsilon: {epsilon:.4f} | Steps: {steps_done}")
        print(f"Exploration: {exploration_actions} actions ({explore_ratio:.1f}%)")
        print(f"Exploitation: {exploitation_actions} actions ({exploit_ratio:.1f}%)")
        print(f"Decay Progress: {decay_progress:.1f}%")
        print(f"{'='*60}\n")
        if score > max_score:
            max_score = int(score)
        torch.save(net.state_dict(), MODEL_PATH)
        try:
            with open(STEPS_PATH, 'w') as f:
                json.dump({
                    'steps_done': steps_done,
                    'generation': generation,
                    'cumulative_score': cumulative_score,
                    'max_score': max_score,
                    'start_time': start_time
                }, f)
            print(f"Saved stats to {STEPS_PATH}")
        except Exception as e:
            print(f"Failed to save stats to {STEPS_PATH}: {e}")
        print(f"Model saved to {MODEL_PATH} (game over)")
        active_color = random.choice(BASE_COLORS)
        next_color = random.choice(BASE_COLORS)
        spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, box_size, color=active_color)
        next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, box_size, color=next_color)
        simulation = Simulation(PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT, CELL_SIZE)
        tetrominoes = []
        active_tetromino = SandTetromino(spawn_x, spawn_y, box_size, color=active_color)
        tetrominoes.append(active_tetromino)
        next_piece = SandTetromino(next_spawn_x, spawn_y, box_size, color=next_color)
        prev_broken = False
        cumulative_score += score
        score = 0
        level = 1
        pixels_cleared_total = 0
        generation += 1
        last_state = prepare_nn_input(simulation.grid)
        last_score = score
        nn_soft_drop_frame_counter = 0
        last_reward_components['score']=0
        soft_drop_frame_counter = 0
        game_over = False
        live=0
        generation_start_time = time.time()
        exploration_actions = 0
        exploitation_actions = 0
        _text_cache.clear()
        _label_cache.clear()


    train_frame_counter += 1
    ai_action_this_frame = False
    if train_frame_counter % ACTION_INTERVAL == 1:
        nn_input = prepare_nn_input(simulation.grid).to(device, non_blocking=True)
        epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
        steps_done += 1
        
        if random.random() < epsilon:
            action_idx = random.randint(0, N_ACTIONS - 1)
            exploration_actions += 1
        else:
            with torch.no_grad(), nullcontext():
                q_values = net(nn_input)
                action_idx = q_values.argmax().item()
            exploitation_actions += 1
        
        action = ACTIONS[action_idx]
        last_action = action
        last_action_idx = action_idx
        ai_action_this_frame = True
        action_str = f"Action: {action}"
        add_terminal_line(action_str)
    else:
        action = last_action
        action_idx = last_action_idx
        ai_action_this_frame = False

    base_gravity = get_gravity(level)
    if ai_action_this_frame:
        can_rotate = all((box.y // CELL_SIZE) >= GHOST_ROWS for box in active_tetromino.boxes)
        if action == 'hold_left':
            move_left = True
            move_right = False
        elif action == 'hold_right':
            move_right = True
            move_left = False
        elif action == 'hold_drop':
            speed_down=True
        elif action in ['left', 'right']:
            move_left = move_right = speed_down = False
            if action == 'left':
                active_tetromino.x -= move_speed
            elif action == 'right':
                active_tetromino.x += move_speed
        elif action == 'rotate_right' and can_rotate:
            active_tetromino.rotate('right', wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS, grid=simulation.grid)
        elif action == 'rotate_left' and can_rotate:
            active_tetromino.rotate('left', wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS, grid=simulation.grid)
        elif action == 'drop':
            active_tetromino.gravity = base_gravity * 2.5
            nn_soft_drop_frame_counter += 1
            if nn_soft_drop_frame_counter % 2 == 0:
                score += 1
        elif action == 'none':
            nn_soft_drop_frame_counter = 0
            pass
        else:
            nn_soft_drop_frame_counter = 0
    MOVE_HOLD_INTERVAL = 1
    move_hold_counter += 1
    if move_hold_counter >= MOVE_HOLD_INTERVAL:
        if move_left:
            active_tetromino.x -= move_speed
        if move_right:
            active_tetromino.x += move_speed
        move_hold_counter = 0

    min_dx = min(dx for dx, dy in active_tetromino.offsets)
    max_dx = max(dx for dx, dy in active_tetromino.offsets)

    leftmost_pos = active_tetromino.x + min_dx * box_size
    rightmost_pos = active_tetromino.x + (max_dx + 1) * box_size

    if leftmost_pos < WALL_THICKNESS:
        active_tetromino.x = WALL_THICKNESS - min_dx * box_size
    if rightmost_pos > PLAYFIELD_WIDTH - WALL_THICKNESS:
        active_tetromino.x = PLAYFIELD_WIDTH - WALL_THICKNESS - (max_dx + 1) * box_size

    if speed_down:
        effective_gravity = base_gravity * 2
        soft_drop_frame_counter += 1
        if soft_drop_frame_counter % 2 == 0:
            score += 1
    else:
        effective_gravity = base_gravity
        soft_drop_frame_counter = 0

    active_tetromino.gravity = effective_gravity

    for tetromino in tetrominoes:
        if not tetromino.broken:
            tetromino.update(simulation, ground_y, wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS)
    simulation.update()

    total_pixels_cleared = 0
    sand_clear_score_this_frame = 0
    while True:
        cleared = simulation.grid.flood_fill_clear(0, simulation.grid.columns - 1)
        if cleared == 0:
            break
        total_pixels_cleared += cleared

    if total_pixels_cleared > 0:
        points = get_points_for_pixels(total_pixels_cleared)
        sand_clear_score_this_frame = points
        score_str = f"cleared {total_pixels_cleared} +{int(points)} (score {int(score)}→{int(score+points)})"
        add_terminal_line(score_str, highlight=True)
        print(f"cleared {total_pixels_cleared} pixels +{points} points (score was {score} -> {score + points})")
        score += points
        pixels_cleared_total += total_pixels_cleared
        level = get_level(score)

    just_broke = (not prev_broken) and active_tetromino.broken
    prev_broken = active_tetromino.broken

    if active_tetromino.broken:
        grid = simulation.grid
        sand_in_ghost = any(
            grid.cells[row][col] is not None
            for row in range(GHOST_ROWS)
            for col in range(grid.columns)
        )
        if sand_in_ghost:
            game_over = True
            continue
        active_tetromino = next_piece
        next_color = random.choice(BASE_COLORS)
        next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, box_size, color=next_color)
        next_piece = SandTetromino(next_spawn_x, spawn_y, box_size, color=next_color)
        tetrominoes.append(active_tetromino)
    active_tetromino.gravity = get_gravity(level)

    current_score = score
    score_diff = current_score - last_score
    last_score = current_score
    
    grid_height = simulation.grid.rows
    color_index = build_color_index(simulation.grid)
    sand_height = 0
    for row in range(color_index.shape[0]):
        if np.any(color_index[row] > 0):
            sand_height = color_index.shape[0] - row
            break

    reward = calculate_reward(color_index, score_diff, game_over)
    cumulative_reward += reward
    next_state = prepare_nn_input(simulation.grid)
    done = game_over

    if ai_action_this_frame:
        replay_buffer.append((last_state, last_action_idx, reward, next_state, done))

    last_reward = reward
    last_state = next_state

    window.fill(THEME_COLOR)
    pygame.draw.rect(window, PLAYFIELD_BG_COLOR, (PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT))
    playfield_surface = window.subsurface((PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT))
    simulation.draw(playfield_surface)
    for tetromino in tetrominoes:
        for box in tetromino.boxes:
            if 0 <= box.y < PLAYFIELD_HEIGHT and 0 <= box.x < PLAYFIELD_WIDTH:
                box.draw(playfield_surface)
    ghost_line_y = VISIBLE_WINDOW_Y - 1
    dot_color = (255, 248, 220)
    dot_length = 8
    gap_length = 8
    x = 0
    while x < PLAYFIELD_WIDTH:
        pygame.draw.line(playfield_surface, dot_color, (x, ghost_line_y), (min(x+dot_length, PLAYFIELD_WIDTH-1), ghost_line_y), 2)
        x += dot_length + gap_length
    sidebar_x = PLAYFIELD_X + PLAYFIELD_WIDTH
    sidebar_w = SIDEBAR_WIDTH
    pygame.draw.rect(window, (40, 40, 40), (sidebar_x, 0, sidebar_w, WINDOW_HEIGHT))
    left_sidebar_w = LEFT_SIDEBAR_WIDTH
    if left_sidebar_w > 0:
        pygame.draw.rect(window, (40, 40, 40), (0, 0, left_sidebar_w, WINDOW_HEIGHT))
        font_term = pygame.font.SysFont(None, 22)
        y0 = 24
        line_height = 26
        max_line_width = left_sidebar_w - 24
        for i, line in enumerate(terminal_lines):
            color = (80, 255, 80) if terminal_highlight[i] else (220, 220, 220)
            words = line.split(' ')
            current = ''
            y = y0
            for word in words:
                test = current + (' ' if current else '') + word
                surf = font_term.render(test, True, color)
                if surf.get_width() > max_line_width:
                    surf2 = font_term.render(current, True, color)
                    window.blit(surf2, (12, y))
                    y += line_height
                    current = word
                else:
                    current = test
            if current:
                surf2 = font_term.render(current, True, color)
                window.blit(surf2, (12, y))
            y0 = y + line_height
    font = pygame.font.SysFont(None, 32, bold=True)
    gen_text = font.render(f"Generation {generation}", True, (255,255,255))
    gen_rect = gen_text.get_rect(topleft=(sidebar_x + 24, 30))
    window.blit(gen_text, gen_rect)
    level_text = font.render(f"Level: {level}", True, (255,255,255))
    level_rect = level_text.get_rect(topleft=(sidebar_x + 24, 70))
    window.blit(level_text, level_rect)
    score_text = font.render(f"Score: {int(score)}", True, (255,255,255))
    score_rect = score_text.get_rect(topleft=(sidebar_x + 24, 110))
    window.blit(score_text, score_rect)
    max_score_text = font.render(f"Max: {int(max_score)}", True, (255,255,255))
    max_score_rect = max_score_text.get_rect(topleft=(sidebar_x + 24, 150))
    window.blit(max_score_text, max_score_rect)
    if(generation!=1):
        avg_score_text = font.render(f"Avg: {int(cumulative_score//(generation-1))}", True, (255,255,255))
        avg_score_rect = avg_score_text.get_rect(topleft=(sidebar_x + 24, 190))
        window.blit(avg_score_text, avg_score_rect)
    gen_elapsed = int(time.time() - generation_start_time)
    gen_hours, gen_rem = divmod(gen_elapsed, 3600)
    gen_mins, gen_secs = divmod(gen_rem, 60)
    gen_time_text = font.render(f"Time:  {gen_hours:02}:{gen_mins:02}:{gen_secs:02}", True, (255,255,255))
    gen_time_rect = gen_time_text.get_rect(topleft=(sidebar_x + 24, 230))
    window.blit(gen_time_text, gen_time_rect)
    total_elapsed = int(time.time() - start_time)
    total_hours, total_rem = divmod(total_elapsed, 3600)
    total_mins, total_secs = divmod(total_rem, 60)
    total_time_text = font.render(f"Total: {total_hours:02}:{total_mins:02}:{total_secs:02}", True, (255,255,255))
    total_time_rect = total_time_text.get_rect(topleft=(sidebar_x + 24, 270))
    window.blit(total_time_text, total_time_rect)
    preview_text = font.render("Next:", True, (255,255,255))
    preview_rect = preview_text.get_rect(topleft=(sidebar_x + 24, 330))
    window.blit(preview_text, preview_rect)
    min_bx = min(box.x for box in next_piece.boxes)
    min_by = min(box.y for box in next_piece.boxes)
    max_bx = max(box.x + box.size for box in next_piece.boxes)
    max_by = max(box.y + box.size for box in next_piece.boxes)
    piece_width = max_bx - min_bx
    piece_height = max_by - min_by
    preview_left = sidebar_x + 24
    preview_top = 380
    offset_x = preview_left - min_bx
    offset_y = preview_top - min_by
    for box in next_piece.boxes:
        box.draw(window, x=box.x + offset_x, y=box.y + offset_y)
    reward_color = (80, 255, 80) if last_reward >= 0 else (255, 80, 80)
    reward_text = font.render(f"Reward: {last_reward:.2f}", True, reward_color)
    reward_rect = reward_text.get_rect(topleft=(sidebar_x + 24, 310))
    window.blit(reward_text, reward_rect)
    y_rb = 600
    for name, val in last_reward_components.items():
        sign = '+' if val >= 0 else ''
        txt = f"{name}: {sign}{val:.1f}"
        surf = RULES_FONT.render(txt, True, (200,200,200))
        window.blit(surf, (sidebar_x + 24, y_rb))
        y_rb += surf.get_height() + 4

    font_rules = pygame.font.SysFont(None, 24, bold=True)
    rules_x = 16
    max_rule_width = LEFT_SIDEBAR_WIDTH - 32 if LEFT_SIDEBAR_WIDTH > 0 else 220
    line_height = 24
    total_lines = 0
    for rule in AI_RULESET:
        words = rule.split(' ')
        current = ''
        for word in words:
            test = current + (' ' if current else '') + word
            surf = font_rules.render(test, True, (255, 60, 60))
            if surf.get_width() > max_rule_width:
                total_lines += 1
                current = word
            else:
                current = test
        if current:
            total_lines += 1
    rules_block_height = total_lines * line_height
    rules_y = (WINDOW_HEIGHT - rules_block_height) // 2
    y_offset = rules_y
    for rule in AI_RULESET:
        words = rule.split(' ')
        current = ''
        for word in words:
            test = current + (' ' if current else '') + word
            surf = font_rules.render(test, True, (255, 60, 60))
            if surf.get_width() > max_rule_width:
                surf2 = font_rules.render(current, True, (255, 60, 60))
                window.blit(surf2, (rules_x, y_offset))
                y_offset += line_height
                current = word
            else:
                current = test
        if current:
            surf2 = font_rules.render(current, True, (255, 60, 60))
            window.blit(surf2, (rules_x, y_offset))
            y_offset += line_height

    pygame.display.flip()
    clock.tick(FPS)


torch.save(net.state_dict(), MODEL_PATH)

try:
    with open(STEPS_PATH, 'w') as f:
        json.dump({
            'steps_done': steps_done,
            'generation': generation,
            'cumulative_score': cumulative_score,
            'max_score': max_score,
            'start_time': start_time
        }, f)
    print(f"Saved stats to {STEPS_PATH}")
except Exception as e:
    print(f"Failed to save stats to {STEPS_PATH}: {e}")

try:
    with _loss_accum_lock:
        if _loss_accum:
            avg_loss = sum(_loss_accum) / len(_loss_accum)
            loss_history.append(avg_loss)
            _loss_accum.clear()
        with open(LOSS_HISTORY_PATH, 'w') as f:
            json.dump(loss_history, f)
        print(f"Saved loss_history (len={len(loss_history)}) to {LOSS_HISTORY_PATH}")
except Exception as e:
    print(f"Failed to save loss_history: {e}")

torch.save(net.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH} (final)")

pygame.quit()