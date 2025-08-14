import pygame, random
from grid import Grid
from particle import SandParticle

class Simulation:
    def __init__(self, width, height, cell_size):
        """Initialize simulation with grid dimensions and cell size."""
        self.grid = Grid(width, height, 3)
        self.cell_size = 3

    def draw(self, window):
        """Draw the simulation grid on the given window."""
        self.grid.draw(window)

    def remove_particle(self, row, column):
        """Remove particle at the specified grid position."""
        self.grid.remove_particle(row, column)

    def update(self):
        """Update all sand particles in the simulation using alternating column scan."""
        for row in range(self.grid.rows - 2, -1, -1):
            if row % 2 == 0:
                column_range = range(self.grid.columns)
            else:
                column_range = reversed(range(self.grid.columns))

            for column in column_range:
                particle = self.grid.get_cell(row, column)
                if isinstance(particle, SandParticle):
                    new_pos = particle.update(self.grid, row, column)
                    if new_pos != (row, column):
                        self.grid.set_cell(new_pos[0], new_pos[1], particle)
                        self.grid.remove_particle(row, column)

    def restart(self):
        """Clear the simulation grid to restart."""
        self.grid.clear()

    def spawn_sand(self, column, row, color=None, base_color=None):
        """Spawn a sand particle at grid location with optional color."""
        if color is not None:
            self.grid.add_particle_with_color(row, column, color, base_color=base_color)
        else:
            # Fallback to create a basic sand particle
            from particle import SandParticle
            if 0 <= row < self.grid.rows and 0 <= column < self.grid.columns and self.grid.is_cell_empty(row, column):
                particle = SandParticle()
                self.grid.set_cell(row, column, particle)