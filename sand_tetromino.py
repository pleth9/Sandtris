import random
from sand_box import SandBox

TETROMINO_SHAPES = {
    'I': [(0,0), (1,0), (2,0), (3,0)],  # Line piece
    'O': [(0,0), (1,0), (0,1), (1,1)],  # Square piece
    'T': [(0,0), (1,0), (2,0), (1,1)],  # T-shaped piece
    'S': [(1,0), (2,0), (0,1), (1,1)],  # S-shaped piece
    'Z': [(0,0), (1,0), (1,1), (2,1)],  # Z-shaped piece
    'J': [(0,0), (0,1), (1,1), (2,1)],  # J-shaped piece
    'L': [(2,0), (0,1), (1,1), (2,1)]   # L-shaped piece
}

TETROMINO_BASE_COLORS = [
    (255, 0, 0),    # Red
    (255, 255, 0),  # Yellow
    (0, 128, 255),  # Blue
    (0, 200, 0)     # Green
]

SRS_ROTATION_CENTERS = {
    'I': (1.5, 0.5),
    'O': (0.5, 0.5),
    'T': (1, 0),
    'S': (1, 1),
    'Z': (1, 1),
    'J': (1, 1),
    'L': (1, 1),
}

SRS_SPAWN_OFFSETS = {
    'I': (3, -1),
    'O': (4, 0),
    'T': (3, 0),
    'S': (3, 0),
    'Z': (3, 0),
    'J': (3, 0),
    'L': (3, 0),
}

JLSTZ_WALL_KICKS = {
    (0, 1): [(0,0), (-1,0), (-1,1), (0,-2), (-1,-2), (-2,0), (1,0), (-2,1), (1,1)],
    (1, 0): [(0,0), (1,0), (1,-1), (0,2), (1,2), (2,0), (-1,0), (2,-1), (-1,-1)],
    (1, 2): [(0,0), (1,0), (1,-1), (0,2), (1,2), (2,0), (-1,0), (2,-1), (-1,-1)],
    (2, 1): [(0,0), (-1,0), (-1,1), (0,-2), (-1,-2), (-2,0), (1,0), (-2,1), (1,1)],
    (2, 3): [(0,0), (1,0), (1,1), (0,-2), (1,-2), (2,0), (-1,0), (2,1), (-1,1)],
    (3, 2): [(0,0), (-1,0), (-1,-1), (0,2), (-1,2), (-2,0), (1,0), (-2,-1), (1,-1)],
    (3, 0): [(0,0), (-1,0), (-1,-1), (0,2), (-1,2), (-2,0), (1,0), (-2,-1), (1,-1)],
    (0, 3): [(0,0), (1,0), (1,1), (0,-2), (1,-2), (2,0), (-1,0), (2,1), (-1,1)],
}

I_WALL_KICKS = {
    (0, 1): [(0,0), (-2,0), (1,0), (-2,-1), (1,2), (-3,0), (2,0), (-1,0), (3,0)],
    (1, 0): [(0,0), (2,0), (-1,0), (2,1), (-1,-2), (3,0), (-2,0), (1,0), (-3,0)],
    (1, 2): [(0,0), (-1,0), (2,0), (-1,2), (2,-1), (-2,0), (3,0), (1,0), (-3,0)],
    (2, 1): [(0,0), (1,0), (-2,0), (1,-2), (-2,1), (2,0), (-3,0), (-1,0), (3,0)],
    (2, 3): [(0,0), (2,0), (-1,0), (2,1), (-1,-2), (3,0), (-2,0), (1,0), (-3,0)],
    (3, 2): [(0,0), (-2,0), (1,0), (-2,-1), (1,2), (-3,0), (2,0), (-1,0), (3,0)],
    (3, 0): [(0,0), (1,0), (-2,0), (1,-2), (-2,1), (2,0), (-3,0), (-1,0), (3,0)],
    (0, 3): [(0,0), (-1,0), (2,0), (-1,2), (2,-1), (-2,0), (3,0), (1,0), (-3,0)],
}

class SandTetromino:
    def __init__(self, x, y, box_size, shape=None, color=None, base_color=None):
        """Initialize a sand tetromino with position, size, and optional shape/color."""
        if shape is None:
            shape = random.choice(list(TETROMINO_SHAPES.keys()))
        if base_color is None:
            base_color = random.choice(TETROMINO_BASE_COLORS)
        if color is None:
            color = base_color
        self.shape = shape
        self.box_size = box_size
        x_offset, y_offset = SRS_SPAWN_OFFSETS[shape]
        self.x = x + x_offset * box_size
        self.y = y + y_offset * box_size
        self.offsets = TETROMINO_SHAPES[shape][:]
        self.color = color
        self.base_color = base_color
        self.boxes = [SandBox(self.x + dx*box_size, self.y + dy*box_size, box_size, base_color=base_color) for dx, dy in self.offsets]
        self.vy = 0
        self.gravity = 5
        self.broken = False
        self.rotation = 0
        self.freeze_timer = 0

    def update(self, simulation, ground_y, wall_left=0, wall_right=800):
        """Updates tetromino physics and breaks when colliding with ground or sand."""
        if self.broken:
            return
        self.vy = self.gravity
        self.y += self.vy
        for i, (dx, dy) in enumerate(self.offsets):
            self.boxes[i].x = self.x + dx*self.box_size
            self.boxes[i].y = self.y + dy*self.box_size
        for box in self.boxes:
            box_bottom = box.y + box.size
            box_left = int(box.x)
            box_right = int(box.x + box.size)
            if box_bottom >= ground_y:
                self.break_all(simulation)
                return
            if box_left < wall_left or box_right > wall_right:
                if box_left < wall_left:
                    self.x += wall_left - box_left
                if box_right > wall_right:
                    self.x -= box_right - wall_right
                continue
            grid_left = int(box.x // simulation.cell_size)
            grid_right = int((box.x + box.size) // simulation.cell_size)
            box_row = int(box_bottom // simulation.cell_size)
            for col in range(grid_left, grid_right + 1):
                if 0 <= box_row < simulation.grid.rows and 0 <= col < simulation.grid.columns:
                    particle = simulation.grid.get_cell(box_row, col)
                    if particle is not None:
                        self.break_all(simulation)
                        return

    def break_all(self, simulation):
        """Breaks all sand boxes and releases sand particles into simulation."""
        self.broken = True
        all_positions = []
        for box in self.boxes:
            if not box.broken:
                box.broken = True
                positions = box.release_sand(simulation)
                all_positions.extend(positions)
        self.last_released_positions = all_positions

    def draw(self, surface):
        """Draws all sand boxes on the given surface."""
        for box in self.boxes:
            box.draw(surface)

    def rotate(self, direction='right', wall_left=0, wall_right=800, grid=None):
        """Rotates tetromino using SRS rotation system with wall kicks."""
        if self.shape == 'O':
            return

        state = self.rotation % 4
        new_state = (state + (1 if direction=='right' else -1)) % 4

        cx, cy = SRS_ROTATION_CENTERS[self.shape]
        new_offsets = []
        for dx, dy in self.offsets:
            rx, ry = dx - cx, dy - cy
            if direction=='right':
                nx, ny =  ry, -rx
            else:
                nx, ny = -ry,  rx
            new_offsets.append((int(round(cx + nx)), int(round(cy + ny))))

        kicks = I_WALL_KICKS if self.shape=='I' else JLSTZ_WALL_KICKS
        kick_tests = kicks.get((state, new_state), [(0,0)])
        
        extended_kicks = list(kick_tests)
        
        if (state, new_state) in kick_tests:
            for extra_x in [-2, 2, -3, 3]:
                if (extra_x, 0) not in extended_kicks:
                    extended_kicks.append((extra_x, 0))
            for extra_x in [-1, 1, -2, 2]:
                for extra_y in [-1, 1]:
                    if (extra_x, extra_y) not in extended_kicks:
                        extended_kicks.append((extra_x, extra_y))
        
        for idx, (kx, ky) in enumerate(extended_kicks):
            tx = self.x + kx * self.box_size
            ty = self.y - ky * self.box_size

            if self.is_position_valid(new_offsets, tx, ty, wall_left, wall_right, grid):
                self.offsets = new_offsets
                self.x = tx
                self.y = ty
                self.rotation = new_state
                self.sync_boxes()
                return True
                
        return False

    def is_position_valid(self, offsets, test_x, test_y, wall_left, wall_right, grid):
        """Checks if tetromino can exist at given position without collisions."""
        for dx, dy in offsets:
            mino_x = test_x + dx * self.box_size
            mino_y = test_y + dy * self.box_size
            
            mino_left = mino_x
            mino_right = mino_x + self.box_size
            
            if mino_left < wall_left or mino_right > wall_right:
                return False
            
            if mino_y < 0:
                continue
                
            if grid:
                col = int(mino_x // grid.cell_size)
                row = int(mino_y // grid.cell_size)
                
                if row >= grid.rows:
                    return False
                    
                if 0 <= row < grid.rows and 0 <= col < grid.columns:
                    if grid.get_cell(row, col) is not None:
                        return False
        
        return True

    def sync_boxes(self):
        """Updates sand box positions to match current tetromino position."""
        for i, (dx, dy) in enumerate(self.offsets):
            self.boxes[i].x = self.x + dx * self.box_size
            self.boxes[i].y = self.y + dy * self.box_size