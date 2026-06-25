"""Explicit observation encoding for headless Sandtris agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sandtris.core.constants import BASE_COLORS, get_color_index
from sandtris.core.scoring import get_gravity
from sandtris.core.tetromino import SandTetromino, TETROMINO_SHAPES

SHAPES: tuple[str, ...] = tuple(TETROMINO_SHAPES)
SHAPE_TO_ID: dict[str, int] = {shape: index + 1 for index, shape in enumerate(SHAPES)}
ID_TO_SHAPE: dict[int, str] = {index: shape for shape, index in SHAPE_TO_ID.items()}
N_SHAPE_IDS = len(SHAPES) + 1
N_COLOR_IDS = len(BASE_COLORS) + 1
ACTIVE_PIECE_FEATURES = 5
NEXT_PIECE_FEATURES = 2
SCALAR_FEATURES = 4


@dataclass(frozen=True)
class SandtrisObservation:
    """Complete agent-visible state.

    ``board`` stores settled sand only, with 0 for empty and 1..N for base
    colors. ``active_mask`` overlays the current falling piece using the same
    color IDs. Piece feature arrays make shape, color, rotation, and position
    explicit even when the piece is still in ghost rows above the board.
    """

    board: np.ndarray
    active_mask: np.ndarray
    active_piece_features: np.ndarray
    next_piece_features: np.ndarray
    scalar_features: np.ndarray


@dataclass(frozen=True)
class ObservationEncoder:
    """Torch encoder for DQN-style models."""

    shape_ids: int = N_SHAPE_IDS
    color_ids: int = N_COLOR_IDS
    scalar_features: int = SCALAR_FEATURES

    @property
    def input_channels(self) -> int:
        """Settled color channels plus active-piece overlay channels."""
        return (self.color_ids - 1) * 2

    @property
    def meta_features(self) -> int:
        return (
            self.shape_ids
            + self.color_ids
            + 4
            + 2
            + self.shape_ids
            + self.color_ids
            + self.scalar_features
        )

    def encode(
        self,
        observations: Sequence[SandtrisObservation],
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations as ``(board_channels, metadata)`` tensors."""
        if not observations:
            raise ValueError("at least one observation is required")

        board = torch.as_tensor(np.stack([obs.board for obs in observations]), device=device, dtype=torch.long)
        active = torch.as_tensor(np.stack([obs.active_mask for obs in observations]), device=device, dtype=torch.long)
        board_channels = torch.cat(
            (
                _color_channels(board, self.color_ids),
                _color_channels(active, self.color_ids),
            ),
            dim=1,
        )

        active_features = torch.as_tensor(
            np.stack([obs.active_piece_features for obs in observations]),
            device=device,
            dtype=torch.float32,
        )
        next_features = torch.as_tensor(
            np.stack([obs.next_piece_features for obs in observations]),
            device=device,
            dtype=torch.float32,
        )
        scalar_features = torch.as_tensor(
            np.stack([obs.scalar_features for obs in observations]),
            device=device,
            dtype=torch.float32,
        )

        rows = max(1.0, float(board.shape[1]))
        cols = max(1.0, float(board.shape[2]))
        metadata = torch.cat(
            (
                _one_hot(active_features[:, 0], self.shape_ids),
                _one_hot(active_features[:, 1], self.color_ids),
                _one_hot(active_features[:, 2] % 4, 4),
                (active_features[:, 3] / cols).unsqueeze(1),
                (active_features[:, 4] / rows).unsqueeze(1),
                _one_hot(next_features[:, 0], self.shape_ids),
                _one_hot(next_features[:, 1], self.color_ids),
                scalar_features,
            ),
            dim=1,
        )
        return board_channels, metadata


DEFAULT_ENCODER = ObservationEncoder()


def build_color_index(grid: object) -> np.ndarray:
    """Return 0-empty, 1..N color IDs for settled sand."""
    cached = grid.get_color_grid()
    return np.where(cached >= 0, cached + 1, 0).astype(np.uint8, copy=True)


def shape_id(tetromino: SandTetromino | None) -> int:
    if tetromino is None:
        return 0
    return SHAPE_TO_ID.get(tetromino.shape, 0)


def color_id(tetromino: SandTetromino | None) -> int:
    if tetromino is None:
        return 0
    index = get_color_index(tetromino.base_color)
    return 0 if index is None else index + 1


def make_observation(
    grid: object,
    active_tetromino: SandTetromino | None,
    next_piece: SandTetromino | None,
    *,
    cell_size: int,
    score: float = 0.0,
    level: int = 1,
    ghost_rows: int = 0,
    gravity: float | None = None,
) -> SandtrisObservation:
    """Build a complete observation from the current headless game state."""
    board = build_color_index(grid)
    active_mask = build_active_mask(active_tetromino, board.shape, cell_size)
    gravity_value = get_gravity(level) if gravity is None else gravity
    return SandtrisObservation(
        board=board,
        active_mask=active_mask,
        active_piece_features=np.array(
            [
                shape_id(active_tetromino),
                color_id(active_tetromino),
                0 if active_tetromino is None else int(active_tetromino.rotation) % 4,
                0 if active_tetromino is None else int(active_tetromino.x // cell_size),
                0 if active_tetromino is None else int(active_tetromino.y // cell_size),
            ],
            dtype=np.int16,
        ),
        next_piece_features=np.array(
            [
                shape_id(next_piece),
                color_id(next_piece),
            ],
            dtype=np.int16,
        ),
        scalar_features=np.array(
            [
                float(score) / 10000.0,
                float(level) / 20.0,
                float(gravity_value) / 10.0,
                float(ghost_rows) / max(1.0, float(board.shape[0])),
            ],
            dtype=np.float32,
        ),
    )


def build_active_mask(
    tetromino: SandTetromino | None,
    board_shape: tuple[int, int],
    cell_size: int,
) -> np.ndarray:
    """Rasterize the falling tetromino into board coordinates."""
    rows, cols = board_shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    piece_color = color_id(tetromino)
    if tetromino is None or piece_color == 0:
        return mask

    for box in tetromino.boxes:
        top = int(box.y // cell_size)
        bottom = int((box.y + box.size - 1) // cell_size)
        left = int(box.x // cell_size)
        right = int((box.x + box.size - 1) // cell_size)
        if bottom < 0 or top >= rows or right < 0 or left >= cols:
            continue
        mask[max(0, top) : min(rows, bottom + 1), max(0, left) : min(cols, right + 1)] = piece_color

    return mask


def _color_channels(ids: torch.Tensor, color_ids: int) -> torch.Tensor:
    channels = [ids == color_index for color_index in range(1, color_ids)]
    return torch.stack(channels, dim=1).to(dtype=torch.float32)


def _one_hot(values: torch.Tensor, width: int) -> torch.Tensor:
    return F.one_hot(values.to(dtype=torch.long).clamp(0, width - 1), num_classes=width).to(dtype=torch.float32)


__all__ = [
    "DEFAULT_ENCODER",
    "ID_TO_SHAPE",
    "N_COLOR_IDS",
    "N_SHAPE_IDS",
    "SHAPES",
    "SHAPE_TO_ID",
    "ObservationEncoder",
    "SandtrisObservation",
    "build_active_mask",
    "build_color_index",
    "color_id",
    "make_observation",
    "shape_id",
]
