"""Deterministic board evaluators and placement scaffolding for Sandtris AI."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from sandtris.ai.actions import PlacementAction
from sandtris.core.tetromino import SandTetromino


@dataclass(frozen=True)
class ComponentStats:
    color_id: int
    size: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int

    @property
    def horizontal_span(self) -> int:
        return self.max_col - self.min_col + 1

    def touches_left(self) -> bool:
        return self.min_col == 0

    def touches_right(self, columns: int) -> bool:
        return self.max_col == columns - 1


@dataclass(frozen=True)
class BoardEvaluation:
    score: float
    max_height: int
    mean_height: float
    danger_cells: int
    bridge_potential: float


@dataclass(frozen=True)
class PlacementEvaluation:
    action: PlacementAction
    score: float
    board_score: float
    bridge_potential: float
    predicted_clear_pixels: int
    predicted_clear_events: int
    max_height: int
    overflow_cells: int


def iter_color_components(board_ids: np.ndarray) -> list[ComponentStats]:
    """Return 4-connected same-color components from a 0-empty board."""
    board = _as_board_ids(board_ids)
    if board.size == 0:
        return []

    rows, cols = board.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components: list[ComponentStats] = []

    for row in range(rows):
        for col in range(cols):
            color = int(board[row, col])
            if color <= 0 or visited[row, col]:
                continue
            positions = _collect_component(board, visited, row, col, color)
            row_values = [position[0] for position in positions]
            col_values = [position[1] for position in positions]
            components.append(
                ComponentStats(
                    color_id=color,
                    size=len(positions),
                    min_row=min(row_values),
                    max_row=max(row_values),
                    min_col=min(col_values),
                    max_col=max(col_values),
                )
            )

    return components


def estimate_wall_bridge_potential(board_ids: np.ndarray) -> float:
    """Score connected same-color components by horizontal bridge potential."""
    board = _as_board_ids(board_ids)
    if board.size == 0:
        return 0.0

    cols = board.shape[1]
    total = 0.0
    for component in iter_color_components(board):
        span_ratio = component.horizontal_span / max(1, cols)
        size_ratio = min(1.0, component.size / max(1, cols))
        touches_left = component.touches_left()
        touches_right = component.touches_right(cols)
        wall_bonus = (0.35 if touches_left else 0.0) + (0.35 if touches_right else 0.0)
        complete_bridge_bonus = 5.0 if touches_left and touches_right else 0.0
        total += (span_ratio**2) * (1.0 + wall_bonus) + size_ratio * 0.25 + complete_bridge_bonus

    return float(total)


def column_heights(board_ids: np.ndarray) -> np.ndarray:
    """Return per-column settled-sand heights."""
    board = _as_board_ids(board_ids)
    if board.size == 0:
        return np.zeros(0, dtype=np.int16)
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=np.int16)
    occupied = board > 0
    top_indices = np.argmax(occupied, axis=0)
    has_sand = np.any(occupied, axis=0)
    heights[has_sand] = rows - top_indices[has_sand]
    return heights


def count_danger_cells(board_ids: np.ndarray, ghost_rows: int = 2) -> int:
    """Count occupied cells in the top danger band."""
    board = _as_board_ids(board_ids)
    if board.size == 0 or ghost_rows <= 0:
        return 0
    return int(np.count_nonzero(board[:ghost_rows] > 0))


def evaluate_board(board_ids: np.ndarray, ghost_rows: int = 2) -> BoardEvaluation:
    """Return a simple deterministic board score for non-neural baselines."""
    heights = column_heights(board_ids)
    max_height = int(np.max(heights)) if heights.size else 0
    mean_height = float(np.mean(heights)) if heights.size else 0.0
    danger_cells = count_danger_cells(board_ids, ghost_rows)
    bridge_potential = estimate_wall_bridge_potential(board_ids)
    score = bridge_potential * 8.0 - max_height * 0.35 - mean_height * 0.15 - danger_cells * 4.0
    return BoardEvaluation(
        score=float(score),
        max_height=max_height,
        mean_height=mean_height,
        danger_cells=danger_cells,
        bridge_potential=bridge_potential,
    )


def choose_heuristic_placement(env: object) -> PlacementAction:
    """Choose a deterministic placement action for the active Sandtris piece."""
    tetromino = getattr(env, "active_tetromino", None)
    config = getattr(env, "config", None)
    if tetromino is None or config is None:
        return PlacementAction(rotation=0, target_column=0)

    actions = enumerate_placement_actions(
        tetromino,
        board_width=config.playfield_width,
        cell_size=config.cell_size,
        box_size=config.box_size,
        wall_left=config.wall_thickness,
        wall_right=config.playfield_width - config.wall_thickness,
    )
    if not actions:
        return PlacementAction(rotation=0, target_column=0)

    best: PlacementEvaluation | None = None
    for action in actions:
        evaluation = evaluate_placement_action(env, action)
        if best is None or evaluation.score > best.score:
            best = evaluation

    return actions[0] if best is None else best.action


def evaluate_placement_action(env: object, action: PlacementAction) -> PlacementEvaluation:
    """Score one placement using a fast deterministic Sandtris approximation."""
    observation = env.observe()
    config = env.config
    board = np.asarray(observation.board, dtype=np.uint8)
    color_id = int(observation.active_piece_features[1]) or 1
    projected_board, overflow_cells = simulate_placement_board(
        board,
        env.active_tetromino,
        action,
        cell_size=config.cell_size,
        box_size=config.box_size,
        color_id=color_id,
    )
    projected_eval = evaluate_board(projected_board, ghost_rows=config.ghost_rows)
    clear_pixels, clear_events, cleared_board = remove_wall_to_wall_bridges(projected_board)
    if clear_pixels:
        cleared_board = apply_column_gravity(cleared_board)
    settled_eval = evaluate_board(cleared_board, ghost_rows=config.ghost_rows)

    score = (
        settled_eval.score
        + projected_eval.bridge_potential * 6.0
        + clear_pixels * 0.50
        + clear_events * 25.0
        - overflow_cells * 10.0
    )
    return PlacementEvaluation(
        action=action,
        score=float(score),
        board_score=settled_eval.score,
        bridge_potential=projected_eval.bridge_potential,
        predicted_clear_pixels=int(clear_pixels),
        predicted_clear_events=int(clear_events),
        max_height=settled_eval.max_height,
        overflow_cells=int(overflow_cells),
    )


def simulate_placement_board(
    board_ids: np.ndarray,
    tetromino: SandTetromino,
    action: PlacementAction,
    *,
    cell_size: int,
    box_size: int,
    color_id: int,
) -> tuple[np.ndarray, int]:
    """Approximate the settled board after dropping a placement action.

    Sandtris boxes shatter into many grid cells. This fast scorer projects
    those cells into their target columns and stacks them downward, which keeps
    the policy deterministic and headless while still optimizing the game's
    wall-to-wall color objective.
    """
    board = np.array(_as_board_ids(board_ids), dtype=np.uint8, copy=True)
    if board.size == 0:
        return board, 0

    box_cells = max(1, int(round(box_size / max(1, cell_size))))
    offsets = _offsets_after_rotations(tetromino, action.rotation)
    min_dx = min(dx for dx, _dy in offsets)
    overflow_cells = 0

    for dx, dy in sorted(offsets, key=lambda offset: (offset[1], offset[0]), reverse=True):
        left_col = action.target_column + (dx - min_dx) * box_cells
        for local_col in range(box_cells):
            col = left_col + local_col
            for _local_row in range(box_cells):
                if not _drop_color_cell(board, col, color_id):
                    overflow_cells += 1

    return board, overflow_cells


def remove_wall_to_wall_bridges(board_ids: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Remove same-color wall bridges from an ID board and report clear size."""
    board = np.array(_as_board_ids(board_ids), dtype=np.uint8, copy=True)
    positions = _wall_to_wall_positions(board)
    if not positions:
        return 0, 0, board

    clear_positions = {position for region in positions for position in region}
    for row, col in clear_positions:
        board[row, col] = 0
    return len(clear_positions), len(positions), board


def apply_column_gravity(board_ids: np.ndarray) -> np.ndarray:
    """Compact non-empty board IDs downward in each column."""
    board = np.array(_as_board_ids(board_ids), dtype=np.uint8, copy=True)
    if board.size == 0:
        return board

    rows, cols = board.shape
    settled = np.zeros_like(board)
    for col in range(cols):
        values = [int(board[row, col]) for row in range(rows) if int(board[row, col]) > 0]
        for index, value in enumerate(reversed(values)):
            settled[rows - 1 - index, col] = value
    return settled


def enumerate_placement_actions(
    tetromino: SandTetromino,
    *,
    board_width: int,
    cell_size: int,
    box_size: int,
    wall_left: int = 0,
    wall_right: int | None = None,
) -> list[PlacementAction]:
    """Enumerate rotation/target-column candidates for placement-level agents."""
    right_wall = board_width if wall_right is None else wall_right
    candidates: set[PlacementAction] = set()

    for rotation in range(4):
        offsets = _offsets_after_rotations(tetromino, rotation)
        min_dx = min(dx for dx, _dy in offsets)
        max_dx = max(dx for dx, _dy in offsets)
        piece_width = (max_dx - min_dx + 1) * box_size
        max_left = right_wall - piece_width
        for left_x in range(int(wall_left), int(max_left) + 1, cell_size):
            candidates.add(PlacementAction(rotation=rotation, target_column=int(round(left_x / cell_size))))

    return sorted(candidates, key=lambda action: (action.rotation, action.target_column))


def _collect_component(
    board: np.ndarray,
    visited: np.ndarray,
    start_row: int,
    start_col: int,
    color: int,
) -> list[tuple[int, int]]:
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited[start_row, start_col] = True
    positions: list[tuple[int, int]] = []

    while queue:
        row, col = queue.popleft()
        positions.append((row, col))
        for row_delta, col_delta in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            next_row = row + row_delta
            next_col = col + col_delta
            if not (0 <= next_row < board.shape[0] and 0 <= next_col < board.shape[1]):
                continue
            if visited[next_row, next_col] or int(board[next_row, next_col]) != color:
                continue
            visited[next_row, next_col] = True
            queue.append((next_row, next_col))

    return positions


def _offsets_after_rotations(tetromino: SandTetromino, rotations: int) -> list[tuple[int, int]]:
    clone = deepcopy(tetromino)
    for _ in range(rotations % 4):
        clone.rotate("right", wall_left=-100000, wall_right=100000, grid=None)
    return clone.offsets


def _drop_color_cell(board: np.ndarray, col: int, color_id: int) -> bool:
    if col < 0 or col >= board.shape[1]:
        return False
    empty_rows = np.flatnonzero(board[:, col] == 0)
    if empty_rows.size == 0:
        return False
    board[int(empty_rows[-1]), col] = color_id
    return True


def _wall_to_wall_positions(board_ids: np.ndarray) -> list[set[tuple[int, int]]]:
    board = _as_board_ids(board_ids)
    if board.size == 0:
        return []

    rows, cols = board.shape
    visited = np.zeros((rows, cols), dtype=bool)
    regions: list[set[tuple[int, int]]] = []

    for row in range(rows):
        for col in range(cols):
            color = int(board[row, col])
            if color <= 0 or visited[row, col]:
                continue
            positions = _collect_component(board, visited, row, col, color)
            col_values = [position[1] for position in positions]
            if min(col_values) == 0 and max(col_values) == cols - 1:
                regions.append(set(positions))

    return regions


def _as_board_ids(board_ids: np.ndarray) -> np.ndarray:
    board = np.asarray(board_ids)
    if board.size and int(np.min(board)) < 0:
        return np.where(board >= 0, board + 1, 0).astype(np.int16, copy=False)
    return board.astype(np.int16, copy=False)


__all__ = [
    "BoardEvaluation",
    "ComponentStats",
    "PlacementEvaluation",
    "apply_column_gravity",
    "choose_heuristic_placement",
    "column_heights",
    "count_danger_cells",
    "enumerate_placement_actions",
    "evaluate_placement_action",
    "estimate_wall_bridge_potential",
    "evaluate_board",
    "iter_color_components",
    "remove_wall_to_wall_bridges",
    "simulate_placement_board",
]
