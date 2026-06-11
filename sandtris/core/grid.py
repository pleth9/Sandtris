"""Grid storage, drawing, and clear operations."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from sandtris.core.clears import ClearRegion, find_wall_to_wall_clears
from sandtris.core.constants import Color, get_color_index, get_gradient_base_color
from sandtris.core.particles import SandParticle


class Grid:
    """Grid storing sand particles by row and column."""

    def __init__(self, width: int, height: int, cell_size: int):
        self.rows = height // cell_size
        self.columns = width // cell_size
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cells: list[list[SandParticle | None]] = [
            [None for _ in range(self.columns)] for _ in range(self.rows)
        ]
        self.highlighted_cells: set[tuple[int, int]] = set()
        self._color_grid_cache: np.ndarray | None = None
        self._cache_dirty = True

    def mark_cache_dirty(self) -> None:
        self._cache_dirty = True

    def get_color_grid(self) -> np.ndarray:
        """Return a cached canonical color-index grid."""
        if self._cache_dirty or self._color_grid_cache is None:
            color_grid = np.full((self.rows, self.columns), -1, dtype=np.int16)
            for row in range(self.rows):
                for col in range(self.columns):
                    particle = self.cells[row][col]
                    if particle is None:
                        continue
                    color_index = get_color_index(get_gradient_base_color(particle))
                    if color_index is not None:
                        color_grid[row, col] = color_index

            self._color_grid_cache = color_grid
            self._cache_dirty = False

        return self._color_grid_cache

    def is_position_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.columns

    def is_cell_empty(self, row: int, col: int) -> bool:
        return self.is_position_valid(row, col) and self.cells[row][col] is None

    def get_cell(self, row: int, col: int) -> SandParticle | None:
        if self.is_position_valid(row, col):
            return self.cells[row][col]
        return None

    get_particle = get_cell

    def set_cell(self, row: int, col: int, particle: SandParticle | None) -> bool:
        if not self.is_position_valid(row, col):
            return False
        self.cells[row][col] = particle
        self.mark_cache_dirty()
        return True

    def add_particle(self, particle: SandParticle) -> bool:
        row = int(getattr(particle, "y", 0) // self.cell_size)
        col = int(getattr(particle, "x", 0) // self.cell_size)
        if self.is_cell_empty(row, col):
            self.cells[row][col] = particle
            self.mark_cache_dirty()
            return True
        return False

    def add_particle_with_color(
        self,
        row: int,
        col: int,
        color: Color,
        base_color: Color | None = None,
    ) -> bool:
        if not self.is_cell_empty(row, col):
            return False
        self.cells[row][col] = SandParticle(color=color, base_color=base_color if base_color else color)
        self.mark_cache_dirty()
        return True

    def remove_particle(self, row: int, col: int) -> SandParticle | None:
        if not self.is_position_valid(row, col):
            return None
        particle = self.cells[row][col]
        if particle is not None:
            self.cells[row][col] = None
            self.mark_cache_dirty()
        return particle

    def move_particle(self, row: int, col: int, next_row: int, next_col: int) -> bool:
        if not self.is_position_valid(row, col) or not self.is_cell_empty(next_row, next_col):
            return False
        particle = self.cells[row][col]
        if particle is None:
            return False
        self.cells[row][col] = None
        self.cells[next_row][next_col] = particle
        self.mark_cache_dirty()
        return True

    def clear(self) -> None:
        self.cells = [[None for _ in range(self.columns)] for _ in range(self.rows)]
        self.highlighted_cells = set()
        self.mark_cache_dirty()

    def find_clear_regions(self) -> list[ClearRegion]:
        return find_wall_to_wall_clears(self)

    def clear_wall_to_wall_regions(self) -> tuple[int, list[ClearRegion]]:
        """Remove all currently clearable regions and return cleared particle count."""
        regions = self.find_clear_regions()
        positions = {position for region in regions for position in region.positions}
        self.highlighted_cells = positions

        for row, col in positions:
            self.remove_particle(row, col)

        return len(positions), regions

    def flood_fill_clear(self, wall_left: int = 0, wall_right: int | None = None) -> int:
        """Compatibility wrapper for the old clear API."""
        cleared, _regions = self.clear_wall_to_wall_regions()
        return cleared

    def flood_fill_clear_from_positions(
        self,
        positions: list[tuple[int, int]] | set[tuple[int, int]],
        min_region_size: int = 1,
    ) -> int:
        """Compatibility wrapper; clear rules remain wall-to-wall only."""
        del positions, min_region_size
        return self.flood_fill_clear()

    def quick_clear_check(self) -> int:
        return self.flood_fill_clear()

    def apply_column_gravity(self) -> None:
        """Compact each column downward after bulk edits or tests."""
        for col in range(self.columns):
            particles = [self.cells[row][col] for row in range(self.rows) if self.cells[row][col] is not None]
            for row in range(self.rows):
                self.cells[row][col] = None
            for index, particle in enumerate(reversed(particles)):
                self.cells[self.rows - 1 - index][col] = particle
        self.mark_cache_dirty()

    _apply_gravity = apply_column_gravity

    def draw(self, window: object) -> None:
        """Draw all particles to a pygame surface."""
        import pygame

        for row in range(self.rows):
            for col in range(self.columns):
                particle = self.cells[row][col]
                if particle is None:
                    continue
                pygame.draw.rect(
                    window,
                    getattr(particle, "color", (200, 200, 200)),
                    (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size),
                )

    def copy(self) -> "Grid":
        new_grid = Grid(self.width, self.height, self.cell_size)
        new_grid.cells = deepcopy(self.cells)
        new_grid.highlighted_cells = set(self.highlighted_cells)
        new_grid.mark_cache_dirty()
        return new_grid
