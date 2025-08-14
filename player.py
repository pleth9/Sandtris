import pygame
from simulation import Simulation
from sand_tetromino import SandTetromino
import time
import random

pygame.init()
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

THEME_COLOR = (40, 40, 40)
PLAYFIELD_BG_COLOR = (0, 0, 0)
WINDOW_HEIGHT = 810
WINDOW_WIDTH = 1440
CELL_SIZE = 4
PLAYFIELD_WIDTH = 492
PLAYFIELD_HEIGHT = 780
PLAYFIELD_X = 450
SIDEBAR_WIDTH = WINDOW_WIDTH - PLAYFIELD_X - PLAYFIELD_WIDTH
LEFT_SIDEBAR_WIDTH = PLAYFIELD_X
FPS = 60
GREY = (29, 29, 29)
WALL_COLOR = (80, 80, 80)
WALL_THICKNESS = 0
GHOST_ROWS = 2
VISIBLE_WINDOW_Y = GHOST_ROWS * CELL_SIZE

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Subscribe to Pleth")
simulation = Simulation(PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT, CELL_SIZE)

ground_y = PLAYFIELD_HEIGHT - 1
box_size = 36
spawn_x = (PLAYFIELD_WIDTH - box_size) // 2
spawn_y = -GHOST_ROWS * box_size

BASE_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 200, 0),
    (255, 255, 0),
]

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
spawn_y = -GHOST_ROWS * box_size
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
pixels_cleared_this_clear = 0
pixels_cleared_total = 0
start_time = time.time()
soft_drop_frame_counter = 0

def get_points_for_pixels(pixels):
    """Calculate score points based on number of pixels cleared with multipliers."""
    if pixels < 100:
        return pixels * 1.0
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

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
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
            elif event.key == pygame.K_r and game_over:
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
                score = 0
                level = 1
                pixels_cleared_total = 0
                soft_drop_frame_counter = 0
                game_over = False
                start_time = time.time()
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
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        window.blit(overlay, (0, 0))
        font = pygame.font.SysFont(None, 60)
        text = font.render("PAUSED", True, (220, 220, 220))
        window.blit(text, ((WINDOW_WIDTH - text.get_width()) // 2, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        clock.tick(FPS)
        continue

    if game_over:
        window.fill(THEME_COLOR)
        simulation.draw(window.subsurface((PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT)))
        for tetromino in tetrominoes:
            tetromino.draw(window.subsurface((PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT)))
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        window.blit(overlay, (0, 0))
        font_large = pygame.font.SysFont(None, 60)
        font_small = pygame.font.SysFont(None, 36)
        game_over_text = font_large.render("GAME OVER", True, (255, 100, 100))
        score_text = font_small.render(f"Final Score: {int(score)}", True, (220, 220, 220))
        restart_text = font_small.render("Press R to restart", True, (220, 220, 220))
        window.blit(game_over_text, ((WINDOW_WIDTH - game_over_text.get_width()) // 2, WINDOW_HEIGHT // 2 - 60))
        window.blit(score_text, ((WINDOW_WIDTH - score_text.get_width()) // 2, WINDOW_HEIGHT // 2))
        window.blit(restart_text, ((WINDOW_WIDTH - restart_text.get_width()) // 2, WINDOW_HEIGHT // 2 + 40))
        pygame.display.flip()
        clock.tick(FPS)
        continue

    if move_left:
        active_tetromino.x -= move_speed
    if move_right:
        active_tetromino.x += move_speed

    min_dx = min(dx for dx, dy in active_tetromino.offsets)
    max_dx = max(dx for dx, dy in active_tetromino.offsets)
    leftmost_pos = active_tetromino.x + min_dx * box_size
    rightmost_pos = active_tetromino.x + (max_dx + 1) * box_size
    if leftmost_pos < WALL_THICKNESS:
        active_tetromino.x = WALL_THICKNESS - min_dx * box_size
    if rightmost_pos > PLAYFIELD_WIDTH - WALL_THICKNESS:
        active_tetromino.x = PLAYFIELD_WIDTH - WALL_THICKNESS - (max_dx + 1) * box_size

    base_gravity = get_gravity(level)

    if speed_down:
        effective_gravity = base_gravity * 2.5
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
    while True:
        cleared = simulation.grid.flood_fill_clear(0, simulation.grid.columns - 1)
        if cleared == 0:
            break
        total_pixels_cleared += cleared

    if total_pixels_cleared > 0:
        points = get_points_for_pixels(total_pixels_cleared)
        print(f"Cleared {total_pixels_cleared} pixels +{points} points (score: {score} -> {score + points})")
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

    font = pygame.font.SysFont(None, 32, bold=True)

    level_text = font.render(f"Level: {level}", True, (255,255,255))
    level_rect = level_text.get_rect(topleft=(sidebar_x + 24, 30))
    window.blit(level_text, level_rect)

    score_text = font.render(f"Score: {int(score)}", True, (255,255,255))
    score_rect = score_text.get_rect(topleft=(sidebar_x + 24, 70))
    window.blit(score_text, score_rect)

    elapsed = int(time.time() - start_time)
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    time_text = font.render(f"Time: {hours:02}:{mins:02}:{secs:02}", True, (255,255,255))
    time_rect = time_text.get_rect(topleft=(sidebar_x + 24, 150))
    window.blit(time_text, time_rect)

    preview_text = font.render("Next:", True, (255,255,255))
    preview_rect = preview_text.get_rect(topleft=(sidebar_x + 24, 210))
    window.blit(preview_text, preview_rect)

    min_bx = min(box.x for box in next_piece.boxes)
    min_by = min(box.y for box in next_piece.boxes)
    max_bx = max(box.x + box.size for box in next_piece.boxes)
    max_by = max(box.y + box.size for box in next_piece.boxes)
    piece_width = max_bx - min_bx
    piece_height = max_by - min_by
    preview_left = sidebar_x + 24
    preview_top = 260
    offset_x = preview_left - min_bx
    offset_y = preview_top - min_by
    for box in next_piece.boxes:
        box.draw(window, x=box.x + offset_x, y=box.y + offset_y)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
