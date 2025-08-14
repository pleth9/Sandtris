import pygame
import numpy as np
from collections import deque

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
    """Return gradient list for a base color."""
    return COLOR_GRADIENTS.get(base_color, [base_color])

def get_gradient_base_color(particle):
    """Get the base color for a particle."""
    return getattr(particle, 'base_color', getattr(particle, 'color', None))

def get_color_index(color):
    """Return index of gradient that contains the given color."""
    if color is None:
        return None
    for idx, (base, gradient) in enumerate(COLOR_GRADIENTS.items()):
        if color in gradient:
            return idx
    return None

class Grid:
    """Grid storing particles and providing flood-fill and drawing utilities."""
    def __init__(self, width, height, cell_size):
        """Initialize the grid with given width, height and cell size."""
        self.rows = height // cell_size
        self.columns = width // cell_size
        self.cell_size = cell_size
        self.cells = [[None for _ in range(self.columns)] for _ in range(self.rows)]
        self._color_grid_cache = None
        self._cache_dirty = True
        
    def _mark_cache_dirty(self):
        """Mark the cached color grid as dirty."""
        self._cache_dirty = True
    
    def _get_color_grid(self):
        """Return a cached color index grid, rebuilding if needed."""
        if self._cache_dirty or self._color_grid_cache is None:
            self._color_grid_cache = np.full((self.rows, self.columns), -1, dtype=np.int8)
            for r in range(self.rows):
                for c in range(self.columns):
                    p = self.cells[r][c]
                    if p is not None:
                        color_idx = get_color_index(get_gradient_base_color(p))
                        if color_idx is not None:
                            self._color_grid_cache[r, c] = color_idx
            self._cache_dirty = False
        return self._color_grid_cache

    def _flood_fill_bfs(self, start_row, start_col, color_idx, color_grid, visited):
        """Fast BFS flood fill for a single connected component."""
        if (start_row < 0 or start_row >= self.rows or 
            start_col < 0 or start_col >= self.columns or
            visited[start_row, start_col] or 
            color_grid[start_row, start_col] != color_idx):
            return []
        
        queue = deque([(start_row, start_col)])
        visited[start_row, start_col] = True
        component = []
        
        touches_left = start_col == 0
        touches_right = start_col == self.columns - 1
        
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            
            # Check if this position touches walls
            if c == 0:
                touches_left = True
            if c == self.columns - 1:
                touches_right = True
            
            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.columns and
                    not visited[nr, nc] and color_grid[nr, nc] == color_idx):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    
                    # Early wall detection
                    if nc == 0:
                        touches_left = True
                    if nc == self.columns - 1:
                        touches_right = True
        
        return component if (touches_left and touches_right) else []

    def flood_fill_clear(self, wall_left, wall_right):
        """Optimized flood fill that only processes necessary regions."""
        color_grid = self._get_color_grid()
        visited = np.zeros((self.rows, self.columns), dtype=bool)
        cleared_positions = set()
        
        # Only check positions that touch the left wall
        for r in range(self.rows):
            if not visited[r, 0] and color_grid[r, 0] >= 0:
                color_idx = color_grid[r, 0]
                component = self._flood_fill_bfs(r, 0, color_idx, color_grid, visited)
                if component:
                    cleared_positions.update(component)
        
        self.highlighted_cells = cleared_positions
        cleared_count = len(cleared_positions)
        
        for r, c in cleared_positions:
            self.remove_particle(r, c)
        
        return cleared_count

    def flood_fill_clear_from_positions(self, positions, min_region_size=1):
        """Optimized version that only checks specific positions."""
        if not positions:
            return 0
            
        color_grid = self._get_color_grid()
        visited = np.zeros((self.rows, self.columns), dtype=bool)
        cleared_positions = set()
        
        for row, col in positions:
            if not (0 <= row < self.rows and 0 <= col < self.columns):
                continue
            if visited[row, col] or color_grid[row, col] < 0:
                continue
                
            color_idx = color_grid[row, col]
            component = self._flood_fill_bfs(row, col, color_idx, color_grid, visited)
            
            if len(component) >= min_region_size:
                cleared_positions.update(component)
        
        self.highlighted_cells = cleared_positions
        for r, c in cleared_positions:
            self.remove_particle(r, c)
        
        return len(cleared_positions)

    def quick_clear_check(self):
        """Super fast check - only look at left wall positions."""
        cleared_positions = set()
        
        # Check each left wall position
        for r in range(self.rows):
            left_particle = self.get_cell(r, 0)
            if left_particle is None:
                continue
                
            color_idx = get_color_index(get_gradient_base_color(left_particle))
            if color_idx is None:
                continue
            
            # Quick horizontal scan to see if we reach the right wall
            current_color = color_idx
            touches_right = False
            connected_positions = [(r, 0)]
            
            # Scan right from this position
            for c in range(1, self.columns):
                particle = self.get_cell(r, c)
                if particle is None:
                    break
                particle_color_idx = get_color_index(get_gradient_base_color(particle))
                if particle_color_idx != current_color:
                    break
                connected_positions.append((r, c))
                if c == self.columns - 1:
                    touches_right = True
            
            # If we have a complete horizontal line, do full flood fill from here
            if touches_right:
                color_grid = self._get_color_grid()
                visited = np.zeros((self.rows, self.columns), dtype=bool)
                component = self._flood_fill_bfs(r, 0, color_idx, color_grid, visited)
                if component:
                    cleared_positions.update(component)
        
        if cleared_positions:
            self.highlighted_cells = cleared_positions
            for r, c in cleared_positions:
                self.remove_particle(r, c)
        
        return len(cleared_positions)

    def clear_highlighted_cells(self):
        """Clear highlighted cells with freeze animation."""
        if not hasattr(self, 'highlighted_cells'):
            return
        # Freeze and color white for a few frames before deleting
        if not hasattr(self, 'clear_freeze_timer'):
            self.clear_freeze_timer = 0
        if self.clear_freeze_timer < 8:
            # Set all highlighted sand to white and freeze
            for r, c in self.highlighted_cells:
                particle = self.get_cell(r, c)
                if particle is not None:
                    particle.color = (255, 255, 255)
                    if not hasattr(particle, 'frozen'):
                        particle.frozen = True
            self.clear_freeze_timer += 1
            return  # Don't delete yet
        # After freeze, delete
        for r, c in self.highlighted_cells:
            self.remove_particle(r, c)
        self.highlighted_cells = set()
        self.clear_freeze_timer = 0
        # Let sand above fall down
        self._apply_gravity()

    def _apply_gravity(self):
        """Optimized gravity application."""
        for col in range(self.columns):
            # Collect all particles in this column
            particles = []
            for row in range(self.rows):
                particle = self.get_cell(row, col)
                if particle is not None:
                    particles.append(particle)
                    self.remove_particle(row, col)
            
            # Place them at the bottom
            for i, particle in enumerate(particles):
                new_row = self.rows - 1 - i
                if new_row >= 0:
                    self.set_cell(new_row, col, particle)

    # Override methods that modify the grid to mark cache dirty
    def add_particle(self, particle):
        """Add a particle to the grid if the cell is empty."""
        row = int(particle.y // self.cell_size)
        col = int(particle.x // self.cell_size)
        
        if self.is_position_valid(row, col) and self.cells[row][col] is None:
            self.cells[row][col] = particle
            self._mark_cache_dirty()
            return True
        return False
    
    def get_particle(self, row, col):
        """Get the particle at the specified row and column, or None if empty."""
        if self.is_position_valid(row, col):
            return self.cells[row][col]
        return None
    
    def remove_particle(self, row, col):
        """Remove and return the particle at the specified position."""
        if self.is_position_valid(row, col):
            particle = self.cells[row][col]
            self.cells[row][col] = None
            self._mark_cache_dirty()
            return particle
        return None
    
    def is_position_valid(self, row, col):
        """Check if the given row and column are within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.columns
    
    def is_cell_empty(self, row, col):
        """Check if the cell at the given position is empty."""
        return self.is_position_valid(row, col) and self.cells[row][col] is None

    def add_particle_with_color(self, row, column, color, base_color=None):
        """Add a particle with specific color to the grid."""
        from particle import SandParticle
        if 0 <= row < self.rows and 0 <= column < self.columns and self.is_cell_empty(row, column):
            particle = SandParticle(color=color, base_color=base_color if base_color else color)
            self.cells[row][column] = particle
            self._mark_cache_dirty()

    def set_cell(self, row, column, particle):
        """Set a particle at the specified position."""
        if 0 <= row < self.rows and 0 <= column < self.columns:
            self.cells[row][column] = particle
            self._mark_cache_dirty()

    def draw(self, window):
        """Draw all particles in the grid to the window."""
        for row in range(self.rows):
            for col in range(self.columns):
                particle = self.cells[row][col]
                if particle is not None:
                    color = getattr(particle, 'color', (200, 200, 200))
                    pygame.draw.rect(window, color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

    def get_cell(self, row, column):
        """Get the particle at the specified position."""
        if 0 <= row < self.rows and 0 <= column < self.columns:
            return self.cells[row][column]
        return None

    def clear(self):
        """Clear all particles from the grid."""
        for row in range(self.rows):
            for column in range(self.columns):
                self.remove_particle(row, column)
        self._mark_cache_dirty()

    def get_full_color_rows(self, wall_left, wall_right):
        """Get rows that are completely filled with the same color."""
        full_rows = []
        left_col = wall_left // self.cell_size
        right_col = (wall_right // self.cell_size) - 1
        for row in range(self.rows):
            colors = []
            for col in range(left_col, right_col + 1):
                particle = self.get_cell(row, col)
                if particle is None:
                    break
                colors.append(particle.color)
            if len(colors) == (right_col - left_col + 1) and all(c == colors[0] for c in colors):
                full_rows.append((row, colors[0]))
        return full_rows

    def highlight_rows(self, rows):
        """Highlight specific rows with their colors."""
        self.highlighted_rows = set(row for row, _ in rows)
        self.highlight_color = {row: color for row, color in rows}

    def clear_highlighted_rows(self):
        """Clear highlighted rows and apply gravity."""
        if not hasattr(self, 'highlighted_rows'):
            return
        for row in self.highlighted_rows:
            for col in range(self.columns):
                self.remove_particle(row, col)
        self.highlighted_rows = set()
        self.highlight_color = {}
        self._apply_gravity()
        self._mark_cache_dirty()
        
    def copy(self):
        """Create a deep copy of the grid."""
        from copy import deepcopy
        new_grid = Grid(self.rows, self.columns, self.cell_size)
        new_grid.cells = deepcopy(self.cells)
        return new_grid