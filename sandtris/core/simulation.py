"""Sand simulation."""

from __future__ import annotations

from sandtris.core.constants import Color
from sandtris.core.grid import Grid
from sandtris.core.particles import SandParticle


class Simulation:
    """Owns the sand grid and advances particle motion."""

    def __init__(self, width: int, height: int, cell_size: int):
        self.grid = Grid(width, height, cell_size)
        self.cell_size = cell_size
        self.frame = 0
        self._active_cells: set[tuple[int, int]] = set()
        self._settled = False
        self._seen_grid_version = self.grid.mutation_version

    def draw(self, window: object) -> None:
        self.grid.draw(window)

    def remove_particle(self, row: int, column: int) -> None:
        self.grid.remove_particle(row, column)
        self.activate_neighborhood(row, column)

    def update(self) -> None:
        """Update sand particles that can plausibly move."""
        self.frame += 1
        if self.grid.mutation_version != self._seen_grid_version and not self._active_cells:
            self._settled = False

        if not self._active_cells:
            self._seen_grid_version = self.grid.mutation_version
            if self.grid.particle_count == 0 or self._settled:
                return
            self.activate_all_particles()

        current_cells = sorted(
            self._active_cells,
            key=lambda position: (position[0], position[1] if (position[0] + self.frame) % 2 == 0 else -position[1]),
            reverse=True,
        )
        self._active_cells = set()

        for row, column in current_cells:
            particle = self.grid.get_cell(row, column)
            if not isinstance(particle, SandParticle):
                continue

            next_row, next_col = particle.update(self.grid, row, column, self.frame)
            if (next_row, next_col) == (row, column):
                if self._can_move_soon(row, column):
                    self._active_cells.add((row, column))
                continue

            if self.grid.move_particle(row, column, next_row, next_col):
                self.activate_neighborhood(row, column)
                self.activate_neighborhood(next_row, next_col)

        self._seen_grid_version = self.grid.mutation_version
        if not self._active_cells:
            self._settled = True

    def restart(self) -> None:
        self.frame = 0
        self.grid.clear()
        self._active_cells.clear()
        self._settled = False
        self._seen_grid_version = self.grid.mutation_version

    def spawn_sand(
        self,
        column: int,
        row: int,
        color: Color | None = None,
        base_color: Color | None = None,
    ) -> bool:
        if color is not None:
            added = self.grid.add_particle_with_color(row, column, color, base_color=base_color)
        else:
            if not self.grid.is_cell_empty(row, column):
                return False
            added = self.grid.set_cell(row, column, SandParticle())

        if added:
            self.activate_neighborhood(row, column)
        return added

    def activate_neighborhood(self, row: int, column: int) -> None:
        """Mark cells affected by movement/removal so stable sand stays asleep."""
        for next_row in range(row - 2, row + 2):
            for next_col in range(column - 1, column + 2):
                if self.grid.is_position_valid(next_row, next_col) and isinstance(self.grid.get_cell(next_row, next_col), SandParticle):
                    self._active_cells.add((next_row, next_col))
        self._settled = False

    def activate_positions(self, positions: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
        for row, column in positions:
            self.activate_neighborhood(row, column)

    def activate_all_particles(self) -> None:
        for row in range(self.grid.rows):
            for column in range(self.grid.columns):
                if isinstance(self.grid.get_cell(row, column), SandParticle):
                    self._active_cells.add((row, column))

    def _can_move_soon(self, row: int, column: int) -> bool:
        return (
            self.grid.is_cell_empty(row + 1, column)
            or self.grid.is_cell_empty(row + 1, column - 1)
            or self.grid.is_cell_empty(row + 1, column + 1)
        )
