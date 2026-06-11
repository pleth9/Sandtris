"""Falling tetromino logic."""

from __future__ import annotations

import random

from sandtris.core.constants import BASE_COLORS, Color
from sandtris.core.sandbox import SandBox

TETROMINO_SHAPES: dict[str, list[tuple[int, int]]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}

TETROMINO_BASE_COLORS = list(BASE_COLORS)

SRS_ROTATION_CENTERS = {
    "I": (1.5, 0.5),
    "O": (0.5, 0.5),
    "T": (1, 0),
    "S": (1, 1),
    "Z": (1, 1),
    "J": (1, 1),
    "L": (1, 1),
}

SRS_SPAWN_OFFSETS = {
    "I": (3, -1),
    "O": (4, 0),
    "T": (3, 0),
    "S": (3, 0),
    "Z": (3, 0),
    "J": (3, 0),
    "L": (3, 0),
}

JLSTZ_WALL_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}

I_WALL_KICKS = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


class SandTetromino:
    """A falling tetromino that breaks into sand on collision."""

    def __init__(
        self,
        x: float,
        y: float,
        box_size: int,
        shape: str | None = None,
        color: Color | None = None,
        base_color: Color | None = None,
    ):
        shape = shape if shape is not None else random.choice(list(TETROMINO_SHAPES))
        base_color = base_color if base_color is not None else random.choice(TETROMINO_BASE_COLORS)
        color = color if color is not None else base_color

        self.shape = shape
        self.box_size = box_size
        x_offset, y_offset = SRS_SPAWN_OFFSETS[shape]
        self.x = x + x_offset * box_size
        self.y = y + y_offset * box_size
        self.offsets = TETROMINO_SHAPES[shape][:]
        self.color = color
        self.base_color = base_color
        self.boxes = [
            SandBox(self.x + dx * box_size, self.y + dy * box_size, box_size, base_color=base_color)
            for dx, dy in self.offsets
        ]
        self.vy = 0.0
        self.gravity = 5.0
        self.broken = False
        self.rotation = 0
        self.last_released_positions: list[tuple[int, int]] = []

    def update(self, simulation: object, ground_y: int, wall_left: int = 0, wall_right: int = 800) -> None:
        if self.broken:
            return

        self.vy = self.gravity
        self.y += self.vy
        self._sync_boxes()
        self._clamp_to_walls(wall_left, wall_right)
        self._sync_boxes()

        if self._collides_with_ground_or_sand(simulation, ground_y):
            self.break_all(simulation)

    def _clamp_to_walls(self, wall_left: int, wall_right: int) -> None:
        min_dx = min(dx for dx, _dy in self.offsets)
        max_dx = max(dx for dx, _dy in self.offsets)
        left = self.x + min_dx * self.box_size
        right = self.x + (max_dx + 1) * self.box_size
        if left < wall_left:
            self.x += wall_left - left
        if right > wall_right:
            self.x -= right - wall_right

    def _collides_with_ground_or_sand(self, simulation: object, ground_y: int) -> bool:
        for box in self.boxes:
            box_bottom = box.y + box.size
            if box_bottom >= ground_y:
                return True

            row_below = int(box_bottom // simulation.cell_size)
            left_col = int(box.x // simulation.cell_size)
            right_col = int((box.x + box.size - 1) // simulation.cell_size)
            for col in range(left_col, right_col + 1):
                if simulation.grid.get_cell(row_below, col) is not None:
                    return True

        return False

    def break_all(self, simulation: object) -> None:
        self.broken = True
        released_positions: list[tuple[int, int]] = []
        for box in self.boxes:
            if not box.broken:
                released_positions.extend(box.release_sand(simulation))
        self.last_released_positions = released_positions

    def draw(self, surface: object) -> None:
        for box in self.boxes:
            box.draw(surface)

    def rotate(
        self,
        direction: str = "right",
        wall_left: int = 0,
        wall_right: int = 800,
        grid: object | None = None,
    ) -> bool:
        if self.shape == "O":
            return False

        state = self.rotation % 4
        new_state = (state + (1 if direction == "right" else -1)) % 4
        cx, cy = SRS_ROTATION_CENTERS[self.shape]
        new_offsets = []

        for dx, dy in self.offsets:
            rel_x, rel_y = dx - cx, dy - cy
            if direction == "right":
                next_x, next_y = rel_y, -rel_x
            else:
                next_x, next_y = -rel_y, rel_x
            new_offsets.append((int(round(cx + next_x)), int(round(cy + next_y))))

        kicks = I_WALL_KICKS if self.shape == "I" else JLSTZ_WALL_KICKS
        for kick_x, kick_y in kicks.get((state, new_state), [(0, 0)]):
            test_x = self.x + kick_x * self.box_size
            test_y = self.y - kick_y * self.box_size
            if self.is_position_valid(new_offsets, test_x, test_y, wall_left, wall_right, grid):
                self.offsets = new_offsets
                self.x = test_x
                self.y = test_y
                self.rotation = new_state
                self._sync_boxes()
                return True

        return False

    def is_position_valid(
        self,
        offsets: list[tuple[int, int]],
        test_x: float,
        test_y: float,
        wall_left: int,
        wall_right: int,
        grid: object | None,
    ) -> bool:
        for dx, dy in offsets:
            mino_x = test_x + dx * self.box_size
            mino_y = test_y + dy * self.box_size
            if mino_x < wall_left or mino_x + self.box_size > wall_right:
                return False
            if mino_y < 0:
                continue
            if grid is not None and self._box_overlaps_grid(mino_x, mino_y, grid):
                return False
        return True

    def _box_overlaps_grid(self, mino_x: float, mino_y: float, grid: object) -> bool:
        top = int(mino_y // grid.cell_size)
        bottom = int((mino_y + self.box_size - 1) // grid.cell_size)
        left = int(mino_x // grid.cell_size)
        right = int((mino_x + self.box_size - 1) // grid.cell_size)
        if bottom >= grid.rows:
            return True
        for row in range(max(0, top), min(grid.rows, bottom + 1)):
            for col in range(max(0, left), min(grid.columns, right + 1)):
                if grid.get_cell(row, col) is not None:
                    return True
        return False

    def sync_boxes(self) -> None:
        self._sync_boxes()

    def _sync_boxes(self) -> None:
        for index, (dx, dy) in enumerate(self.offsets):
            self.boxes[index].x = self.x + dx * self.box_size
            self.boxes[index].y = self.y + dy * self.box_size
