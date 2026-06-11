"""Tetromino block that breaks into sand particles."""

from __future__ import annotations

from math import ceil

from sandtris.core.constants import Color, get_gradient_for_base


class SandBox:
    """A pixel-rendered mino that releases grid-aligned sand."""

    def __init__(self, x: float, y: float, size: int, base_color: Color = (194, 178, 128)):
        self.x = x
        self.y = y
        self.size = size
        self.base_color = base_color
        self.vy = 0.0
        self.gravity = 0.15
        self.broken = False
        self.gradient = get_gradient_for_base(base_color)
        self.sand_colors = self._build_pixel_colors()

    def _build_pixel_colors(self) -> list[list[Color]]:
        n_bands = len(self.gradient)
        colors: list[list[Color]] = [[self.base_color for _ in range(self.size)] for _ in range(self.size)]
        for y in range(self.size):
            for x in range(self.size):
                layer = min(x, y, self.size - 1 - x, self.size - 1 - y)
                band = min(layer * n_bands // max(1, self.size // 2), n_bands - 1)
                colors[y][x] = self.gradient[band]
        return colors

    def release_sand(self, simulation: object) -> list[tuple[int, int]]:
        """Spawn one sand particle per covered grid cell."""
        released_positions: list[tuple[int, int]] = []
        cell_size = simulation.cell_size
        rows = ceil(self.size / cell_size)
        cols = ceil(self.size / cell_size)

        for local_row in range(rows):
            for local_col in range(cols):
                px = int((self.x + local_col * cell_size) // cell_size)
                py = int((self.y + local_row * cell_size) // cell_size)
                if not (0 <= py < simulation.grid.rows and 0 <= px < simulation.grid.columns):
                    continue
                pixel_x = min(local_col * cell_size, self.size - 1)
                pixel_y = min(local_row * cell_size, self.size - 1)
                color = self.sand_colors[pixel_y][pixel_x]
                if simulation.spawn_sand(px, py, color=color, base_color=self.base_color):
                    released_positions.append((py, px))

        self.broken = True
        return released_positions

    def update(self, simulation: object, ground_y: int) -> None:
        if self.broken:
            return

        self.vy += self.gravity
        self.y += self.vy
        if self.y + self.size >= ground_y or self._has_sand_below(simulation):
            self.broken = True
            self.release_sand(simulation)

    def _has_sand_below(self, simulation: object) -> bool:
        cell_size = simulation.cell_size
        row_below = int((self.y + self.size) // cell_size)
        left_col = int(self.x // cell_size)
        right_col = int((self.x + self.size - 1) // cell_size)
        return any(simulation.grid.get_cell(row_below, col) is not None for col in range(left_col, right_col + 1))

    def draw(self, surface: object, x: float | None = None, y: float | None = None) -> None:
        import pygame

        if self.broken:
            return

        draw_x = self.x if x is None else x
        draw_y = self.y if y is None else y
        for pixel_y in range(self.size):
            for pixel_x in range(self.size):
                pygame.draw.rect(
                    surface,
                    self.sand_colors[pixel_y][pixel_x],
                    (int(draw_x + pixel_x), int(draw_y + pixel_y), 1, 1),
                )
