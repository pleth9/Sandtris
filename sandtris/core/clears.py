"""Wall-to-wall clear detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClearRegion:
    """A same-color connected region that touches both side walls."""

    color_index: int
    positions: frozenset[tuple[int, int]]

    @property
    def size(self) -> int:
        return len(self.positions)


def find_wall_to_wall_clears(grid: object) -> list[ClearRegion]:
    """Find 4-connected same-color regions touching both left and right walls."""
    color_grid = grid.get_color_grid()
    if grid.columns == 0 or grid.rows == 0:
        return []

    visited = np.zeros((grid.rows, grid.columns), dtype=bool)
    regions: list[ClearRegion] = []

    for row in range(grid.rows):
        if visited[row, 0] or color_grid[row, 0] < 0:
            continue

        color_index = int(color_grid[row, 0])
        positions, touches_right = _collect_component(grid, color_grid, visited, row, 0, color_index)
        if touches_right:
            regions.append(ClearRegion(color_index=color_index, positions=frozenset(positions)))

    regions.sort(key=lambda region: (min(region.positions), region.color_index))
    return regions


def _collect_component(
    grid: object,
    color_grid: np.ndarray,
    visited: np.ndarray,
    start_row: int,
    start_col: int,
    color_index: int,
) -> tuple[set[tuple[int, int]], bool]:
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited[start_row, start_col] = True
    positions: set[tuple[int, int]] = set()
    touches_right = start_col == grid.columns - 1

    while queue:
        row, col = queue.popleft()
        positions.add((row, col))
        touches_right = touches_right or col == grid.columns - 1

        for row_delta, col_delta in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            next_row = row + row_delta
            next_col = col + col_delta
            if not (0 <= next_row < grid.rows and 0 <= next_col < grid.columns):
                continue
            if visited[next_row, next_col] or int(color_grid[next_row, next_col]) != color_index:
                continue

            visited[next_row, next_col] = True
            queue.append((next_row, next_col))

    return positions, touches_right
