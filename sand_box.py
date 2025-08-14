import pygame
import random
from simulation import Simulation
from particle import SandParticle

COLOR_GRADIENTS = {
    (255, 0, 0): [
        (128, 0, 0), (200, 0, 0), (255, 0, 0), (255, 100, 100)
    ],
    (255, 255, 0): [
        (128, 128, 0), (200, 200, 0), (255, 255, 0), (255, 255, 120)
    ],
    (0, 128, 255): [
        (0, 64, 128), (0, 100, 200), (0, 128, 255), (100, 180, 255)
    ],
    (0, 200, 0): [
        (0, 100, 0), (0, 150, 0), (0, 200, 0), (120, 255, 120)
    ]
}

def get_gradient_for_base(base_color):
    """Returns the color gradient list for the given base color."""
    return COLOR_GRADIENTS.get(base_color, [base_color])

class SandBox:
    def __init__(self, x, y, size, base_color=(194, 178, 128)):
        """Initialize a sand box with position, size, and color."""
        self.x = x
        self.y = y
        self.size = size
        self.base_color = base_color
        self.vy = 0
        self.gravity = 0.15
        self.broken = False
        self.cell_size = 3
        self.sand_particles = [(i, j) for i in range(size) for j in range(size)]
        self.gradient = get_gradient_for_base(base_color)
        n_bands = len(self.gradient)
        self.sand_colors = [[None for _ in range(size)] for _ in range(size)]
        for dx in range(size):
            for dy in range(size):
                layer = min(dx, dy, size - 1 - dx, size - 1 - dy)
                band = min(layer * n_bands // (size // 2), n_bands - 1) if size > 2 else 0
                self.sand_colors[dy][dx] = self.gradient[band]

    def lighten(self, color, factor):
        """Returns a lightened version of the color by the given factor."""
        return tuple(min(int(c + (255 - c) * factor), 255) for c in color)

    def darken(self, color, factor):
        """Returns a darkened version of the color by the given factor."""
        return tuple(max(int(c * (1 - factor)), 0) for c in color)

    def update(self, simulation: Simulation, ground_y):
        """Updates box physics and breaks it when colliding with ground or sand."""
        if not self.broken:
            self.vy += self.gravity
            self.y += self.vy
            box_bottom = self.y + self.size
            box_left = int(self.x // simulation.cell_size)
            box_right = int((self.x + self.size - 1) // simulation.cell_size)
            box_row_below = int((box_bottom) // simulation.cell_size)
            sand_below = False
            for col in range(box_left, box_right + 1):
                if 0 <= box_row_below < simulation.grid.rows and 0 <= col < simulation.grid.columns:
                    particle = simulation.grid.get_cell(box_row_below, col)
                    if particle is not None:
                        sand_below = True
                        break
            if box_bottom > ground_y:
                self.y -= (box_bottom - ground_y)
                self.broken = True
                self.release_sand(simulation)
            elif sand_below:
                self.broken = True
                self.release_sand(simulation)

    def release_sand(self, simulation: Simulation):
        """Spawns sand particles from the box into the simulation grid."""
        released_positions = []
        n_bands = len(self.gradient)
        grid_rows = simulation.grid.rows
        grid_cols = simulation.grid.columns
        for dx, dy in self.sand_particles:
            px = int((self.x + dx) // self.cell_size)
            py = int((self.y + dy) // self.cell_size)
            px = max(0, min(grid_cols - 1, px))
            py = max(0, min(grid_rows - 1, py))
            layer = min(dx, dy, self.size - 1 - dx, self.size - 1 - dy)
            band = min(layer * n_bands // (self.size // 2), n_bands - 1) if self.size > 2 else 0
            color = self.gradient[band]
            simulation.spawn_sand(px, py, color=color, base_color=self.base_color)
            released_positions.append((py, px))
        self.sand_particles = []
        return released_positions

    def break_box(self, simulation):
        """Immediately breaks the box and spawns all sand particles."""
        for dx in range(self.size):
            for dy in range(self.size):
                px = self.x + dx * self.cell_size
                py = self.y + dy * self.cell_size
                color = self.sand_colors[dy][dx]
                simulation.spawn_sand(px, py, color=color, base_color=self.base_color)
        self.broken = True

    def draw(self, surface, x=None, y=None):
        """Draws the sand box on the given surface at specified or current position."""
        draw_x = self.x if x is None else x
        draw_y = self.y if y is None else y
        if not self.broken:
            for dx in range(self.size):
                for dy in range(self.size):
                    color = self.sand_colors[dy][dx]
                    px = int(draw_x + dx)
                    py = int(draw_y + dy)
                    pygame.draw.rect(surface, color, (px, py, 1, 1))