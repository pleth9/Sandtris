"""Playable Sandtris pygame app."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import pygame

from sandtris.core.constants import BASE_COLORS
from sandtris.core.scoring import ScoreState, get_gravity
from sandtris.core.simulation import Simulation
from sandtris.core.tetromino import SandTetromino

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
BOX_SIZE = 36
MOVE_SPEED = 10


@dataclass
class InputState:
    move_left: bool = False
    move_right: bool = False
    speed_down: bool = False


class SandtrisGame:
    """High-level pygame game loop state."""

    def __init__(self):
        pygame.init()
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Sandtris")
        self.input = InputState()
        self.pause = False
        self.game_over = False
        self.running = True
        self.reset()

    def reset(self) -> None:
        self.simulation = Simulation(PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT, CELL_SIZE)
        self.ground_y = PLAYFIELD_HEIGHT - 1
        self.spawn_y = -GHOST_ROWS * BOX_SIZE
        self.score_state = ScoreState()
        self.start_time = time.time()
        self.soft_drop_frame_counter = 0

        active_color = random.choice(BASE_COLORS)
        next_color = random.choice(BASE_COLORS)
        spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, BOX_SIZE, color=active_color)
        next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, BOX_SIZE, color=next_color)
        self.active_tetromino = SandTetromino(spawn_x, self.spawn_y, BOX_SIZE, color=active_color, base_color=active_color)
        self.next_piece = SandTetromino(next_spawn_x, self.spawn_y, BOX_SIZE, color=next_color, base_color=next_color)
        self.tetrominoes = [self.active_tetromino]
        self.game_over = False

    def run(self) -> int:
        while self.running:
            self.handle_events()
            if self.pause:
                self.draw_overlay("PAUSED")
                continue
            if self.game_over:
                self.draw_game_over()
                continue
            self.update()
            self.draw()

        pygame.quit()
        return 0

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_key_down(event.key)
            elif event.type == pygame.KEYUP:
                self.handle_key_up(event.key)

    def handle_key_down(self, key: int) -> None:
        if key == pygame.K_LEFT:
            self.input.move_left = True
        elif key == pygame.K_RIGHT:
            self.input.move_right = True
        elif key == pygame.K_DOWN:
            self.input.speed_down = True
        elif key == pygame.K_UP:
            self.rotate_active("right")
        elif key == pygame.K_z:
            self.rotate_active("left")
        elif key == pygame.K_p:
            self.pause = not self.pause
        elif key == pygame.K_r and self.game_over:
            self.reset()

    def handle_key_up(self, key: int) -> None:
        if key == pygame.K_LEFT:
            self.input.move_left = False
        elif key == pygame.K_RIGHT:
            self.input.move_right = False
        elif key == pygame.K_DOWN:
            self.input.speed_down = False

    def rotate_active(self, direction: str) -> None:
        if all((box.y // CELL_SIZE) >= GHOST_ROWS for box in self.active_tetromino.boxes):
            self.active_tetromino.rotate(
                direction,
                wall_left=WALL_THICKNESS,
                wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS,
                grid=self.simulation.grid,
            )

    def update(self) -> None:
        self.move_active_piece()
        self.apply_gravity()

        for tetromino in self.tetrominoes:
            if not tetromino.broken:
                tetromino.update(
                    self.simulation,
                    self.ground_y,
                    wall_left=WALL_THICKNESS,
                    wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS,
                )

        self.simulation.update()
        self.resolve_clears()
        self.spawn_next_piece_if_needed()

    def move_active_piece(self) -> None:
        if self.input.move_left:
            self.active_tetromino.x -= MOVE_SPEED
        if self.input.move_right:
            self.active_tetromino.x += MOVE_SPEED

        min_dx = min(dx for dx, _dy in self.active_tetromino.offsets)
        max_dx = max(dx for dx, _dy in self.active_tetromino.offsets)
        left = self.active_tetromino.x + min_dx * BOX_SIZE
        right = self.active_tetromino.x + (max_dx + 1) * BOX_SIZE
        if left < WALL_THICKNESS:
            self.active_tetromino.x = WALL_THICKNESS - min_dx * BOX_SIZE
        if right > PLAYFIELD_WIDTH - WALL_THICKNESS:
            self.active_tetromino.x = PLAYFIELD_WIDTH - WALL_THICKNESS - (max_dx + 1) * BOX_SIZE
        self.active_tetromino.sync_boxes()

    def apply_gravity(self) -> None:
        base_gravity = get_gravity(self.score_state.level)
        if self.input.speed_down:
            self.active_tetromino.gravity = base_gravity * 2.5
            self.soft_drop_frame_counter += 1
            if self.soft_drop_frame_counter % 2 == 0:
                self.score_state.add_soft_drop(1)
        else:
            self.active_tetromino.gravity = base_gravity
            self.soft_drop_frame_counter = 0

    def resolve_clears(self) -> None:
        total_cleared = 0
        while True:
            cleared, regions = self.simulation.grid.clear_wall_to_wall_regions()
            if cleared == 0:
                break
            total_cleared += cleared
            self.simulation.activate_positions({position for region in regions for position in region.positions})

        if total_cleared:
            self.score_state.record_clear(total_cleared)

    def spawn_next_piece_if_needed(self) -> None:
        if not self.active_tetromino.broken:
            return

        if self.has_sand_in_ghost_rows():
            self.game_over = True
            return

        self.active_tetromino = self.next_piece
        next_color = random.choice(BASE_COLORS)
        next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, BOX_SIZE, color=next_color)
        self.next_piece = SandTetromino(next_spawn_x, self.spawn_y, BOX_SIZE, color=next_color, base_color=next_color)
        self.tetrominoes.append(self.active_tetromino)
        self.active_tetromino.gravity = get_gravity(self.score_state.level)

    def has_sand_in_ghost_rows(self) -> bool:
        grid = self.simulation.grid
        return any(grid.cells[row][col] is not None for row in range(GHOST_ROWS) for col in range(grid.columns))

    def draw(self) -> None:
        self.window.fill(THEME_COLOR)
        pygame.draw.rect(self.window, PLAYFIELD_BG_COLOR, (PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT))
        playfield = self.window.subsurface((PLAYFIELD_X, 0, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT))
        self.simulation.draw(playfield)
        for tetromino in self.tetrominoes:
            for box in tetromino.boxes:
                if 0 <= box.y < PLAYFIELD_HEIGHT and 0 <= box.x < PLAYFIELD_WIDTH:
                    box.draw(playfield)
        self.draw_ghost_line(playfield)
        self.draw_sidebars()
        pygame.display.flip()
        self.clock.tick(FPS)

    def draw_ghost_line(self, playfield: object) -> None:
        ghost_line_y = VISIBLE_WINDOW_Y - 1
        x = 0
        while x < PLAYFIELD_WIDTH:
            pygame.draw.line(playfield, (255, 248, 220), (x, ghost_line_y), (min(x + 8, PLAYFIELD_WIDTH - 1), ghost_line_y), 2)
            x += 16

    def draw_sidebars(self) -> None:
        sidebar_x = PLAYFIELD_X + PLAYFIELD_WIDTH
        pygame.draw.rect(self.window, THEME_COLOR, (sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(self.window, THEME_COLOR, (0, 0, LEFT_SIDEBAR_WIDTH, WINDOW_HEIGHT))

        font = pygame.font.SysFont(None, 32, bold=True)
        self.blit_text(font, f"Level: {self.score_state.level}", sidebar_x + 24, 30)
        self.blit_text(font, f"Score: {self.score_state.score}", sidebar_x + 24, 70)
        self.blit_text(font, f"Combo: {self.score_state.combo}", sidebar_x + 24, 110)

        elapsed = int(time.time() - self.start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.blit_text(font, f"Time: {hours:02}:{minutes:02}:{seconds:02}", sidebar_x + 24, 160)
        self.blit_text(font, "Next:", sidebar_x + 24, 220)
        self.draw_next_piece(sidebar_x + 24, 270)

    def blit_text(self, font: object, text: str, x: int, y: int) -> None:
        surface = font.render(text, True, (255, 255, 255))
        self.window.blit(surface, surface.get_rect(topleft=(x, y)))

    def draw_next_piece(self, preview_left: int, preview_top: int) -> None:
        min_x = min(box.x for box in self.next_piece.boxes)
        min_y = min(box.y for box in self.next_piece.boxes)
        offset_x = preview_left - min_x
        offset_y = preview_top - min_y
        for box in self.next_piece.boxes:
            box.draw(self.window, x=box.x + offset_x, y=box.y + offset_y)

    def draw_overlay(self, text: str) -> None:
        self.draw()
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.window.blit(overlay, (0, 0))
        font = pygame.font.SysFont(None, 60)
        rendered = font.render(text, True, (220, 220, 220))
        self.window.blit(rendered, ((WINDOW_WIDTH - rendered.get_width()) // 2, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        self.clock.tick(FPS)

    def draw_game_over(self) -> None:
        self.draw()
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.window.blit(overlay, (0, 0))
        large = pygame.font.SysFont(None, 60)
        small = pygame.font.SysFont(None, 36)
        game_over_text = large.render("GAME OVER", True, (255, 100, 100))
        score_text = small.render(f"Final Score: {self.score_state.score}", True, (220, 220, 220))
        restart_text = small.render("Press R to restart", True, (220, 220, 220))
        self.window.blit(game_over_text, ((WINDOW_WIDTH - game_over_text.get_width()) // 2, WINDOW_HEIGHT // 2 - 60))
        self.window.blit(score_text, ((WINDOW_WIDTH - score_text.get_width()) // 2, WINDOW_HEIGHT // 2))
        self.window.blit(restart_text, ((WINDOW_WIDTH - restart_text.get_width()) // 2, WINDOW_HEIGHT // 2 + 40))
        pygame.display.flip()
        self.clock.tick(FPS)


def get_centered_spawn_x(tetromino_class: type[SandTetromino], playfield_width: int, box_size: int, **kwargs: object) -> int:
    temp = tetromino_class(0, 0, box_size, **kwargs)
    min_x = min(box.x for box in temp.boxes)
    max_x = max(box.x + box.size for box in temp.boxes)
    return int((playfield_width - (max_x - min_x)) // 2 - min_x)


def main() -> int:
    return SandtrisGame().run()
