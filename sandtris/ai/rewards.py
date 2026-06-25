"""Centralized reward shaping for Sandtris AI training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sandtris.ai.heuristic import column_heights, count_danger_cells, estimate_wall_bridge_potential


@dataclass(frozen=True)
class RewardConfig:
    step_penalty: float = -0.02
    clear_points_scale: float = 0.10
    mean_height_scale: float = 0.005
    danger_height_ratio: float = 0.70
    danger_height_scale: float = 20.0
    danger_cell_scale: float = 1.0
    bridge_delta_scale: float = 0.50
    bridge_baseline_scale: float = 0.02
    game_over_penalty: float = -50.0
    ghost_rows: int = 2


@dataclass(frozen=True)
class RewardBreakdown:
    step_penalty: float
    clear_reward: float
    height_penalty: float
    danger_penalty: float
    bridge_potential_delta: float
    component_span_reward: float
    game_over_penalty: float
    bridge_potential: float
    total_reward: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_REWARD_CONFIG = RewardConfig()


def calculate_reward(
    board_ids: np.ndarray,
    *,
    clear_points: float,
    game_over: bool,
    previous_bridge_potential: float | None = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> RewardBreakdown:
    """Return inspectable reward components for a post-step board state."""
    board = np.asarray(board_ids)
    bridge_potential = estimate_wall_bridge_potential(board)

    if game_over:
        return RewardBreakdown(
            step_penalty=0.0,
            clear_reward=0.0,
            height_penalty=0.0,
            danger_penalty=0.0,
            bridge_potential_delta=0.0,
            component_span_reward=0.0,
            game_over_penalty=config.game_over_penalty,
            bridge_potential=bridge_potential,
            total_reward=config.game_over_penalty,
        )

    heights = column_heights(board)
    max_height = int(np.max(heights)) if heights.size else 0
    mean_height = float(np.mean(heights)) if heights.size else 0.0
    height_ratio = max_height / max(1, board.shape[0] if board.ndim == 2 else 1)

    clear_reward = float(clear_points) * config.clear_points_scale
    height_penalty = -config.mean_height_scale * mean_height
    danger_penalty = -config.danger_cell_scale * count_danger_cells(board, config.ghost_rows)
    if height_ratio > config.danger_height_ratio:
        scaled = (height_ratio - config.danger_height_ratio) / max(0.001, 1.0 - config.danger_height_ratio)
        danger_penalty -= config.danger_height_scale * scaled**2

    if previous_bridge_potential is None:
        bridge_potential_delta = 0.0
        component_span_reward = bridge_potential * config.bridge_baseline_scale
    else:
        bridge_potential_delta = bridge_potential - previous_bridge_potential
        component_span_reward = bridge_potential_delta * config.bridge_delta_scale

    total = (
        config.step_penalty
        + clear_reward
        + height_penalty
        + danger_penalty
        + component_span_reward
    )
    return RewardBreakdown(
        step_penalty=config.step_penalty,
        clear_reward=clear_reward,
        height_penalty=height_penalty,
        danger_penalty=danger_penalty,
        bridge_potential_delta=bridge_potential_delta,
        component_span_reward=component_span_reward,
        game_over_penalty=0.0,
        bridge_potential=bridge_potential,
        total_reward=float(total),
    )


__all__ = [
    "DEFAULT_REWARD_CONFIG",
    "RewardBreakdown",
    "RewardConfig",
    "calculate_reward",
]
