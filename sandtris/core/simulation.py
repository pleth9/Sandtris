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

    def draw(self, window: object) -> None:
        self.grid.draw(window)

    def remove_particle(self, row: int, column: int) -> None:
        self.grid.remove_particle(row, column)

    def update(self) -> None:
        """Update sand bottom-up with alternating row direction to reduce lateral bias."""
        self.frame += 1
        for row in range(self.grid.rows - 2, -1, -1):
            columns = range(self.grid.columns) if (row + self.frame) % 2 == 0 else range(self.grid.columns - 1, -1, -1)
            for column in columns:
                particle = self.grid.get_cell(row, column)
                if not isinstance(particle, SandParticle):
                    continue
                next_row, next_col = particle.update(self.grid, row, column, self.frame)
                if (next_row, next_col) != (row, column):
                    self.grid.move_particle(row, column, next_row, next_col)

    def restart(self) -> None:
        self.frame = 0
        self.grid.clear()

    def spawn_sand(
        self,
        column: int,
        row: int,
        color: Color | None = None,
        base_color: Color | None = None,
    ) -> bool:
        if color is not None:
            return self.grid.add_particle_with_color(row, column, color, base_color=base_color)
        if not self.grid.is_cell_empty(row, column):
            return False
        return self.grid.set_cell(row, column, SandParticle())
