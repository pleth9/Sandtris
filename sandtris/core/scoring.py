"""Scoring and level progression."""

from __future__ import annotations

from dataclasses import dataclass

LEVEL_SCORE_STEP = 5000


def get_level(score: float) -> int:
    """Return the 1-based level for a score."""
    return max(1, int(score // LEVEL_SCORE_STEP) + 1)


def get_gravity(level: int) -> float:
    """Return falling gravity for a level."""
    return 2.5 * 1.1 ** (level / 5)


def calculate_clear_score(
    pixels: int,
    level: int = 1,
    combo: int = 0,
    soft_drop_points: int = 0,
) -> int:
    """Score a wall-to-wall clear using size, level, combo, and soft-drop bonuses."""
    if pixels <= 0:
        return soft_drop_points

    size_multiplier = _size_multiplier(pixels)
    level_multiplier = 1.0 + (max(1, level) - 1) * 0.08
    combo_bonus = max(0, combo) * 75
    return int(round(pixels * size_multiplier * level_multiplier + combo_bonus + soft_drop_points))


def _size_multiplier(pixels: int) -> float:
    if pixels < 100:
        return 1.0
    if pixels < 400:
        return 1.12
    if pixels < 900:
        return 1.25
    if pixels < 1600:
        return 1.4
    if pixels < 2500:
        return 1.58
    if pixels < 3600:
        return 1.78
    if pixels < 4900:
        return 2.0
    if pixels < 6400:
        return 2.25
    return 2.5


@dataclass
class ScoreState:
    """Mutable score state for a game session."""

    score: int = 0
    level: int = 1
    combo: int = 0
    pixels_cleared_total: int = 0
    soft_drop_points_pending: int = 0
    last_clear_points: int = 0

    def add_soft_drop(self, points: int = 1) -> None:
        self.score += points
        self.soft_drop_points_pending += points
        self.level = get_level(self.score)

    def record_clear(self, pixels: int) -> int:
        self.combo += 1
        points = calculate_clear_score(
            pixels,
            level=self.level,
            combo=self.combo - 1,
            soft_drop_points=self.soft_drop_points_pending,
        )
        self.score += points
        self.pixels_cleared_total += pixels
        self.soft_drop_points_pending = 0
        self.last_clear_points = points
        self.level = get_level(self.score)
        return points

    def record_no_clear(self) -> None:
        self.combo = 0
        self.soft_drop_points_pending = 0
        self.last_clear_points = 0
