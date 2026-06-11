"""Shared constants for Sandtris core modules."""

from __future__ import annotations

from typing import TypeAlias

Color: TypeAlias = tuple[int, int, int]

CELL_SIZE = 3

COLOR_GRADIENTS: dict[Color, list[Color]] = {
    (255, 0, 0): [
        (128, 0, 0),
        (200, 0, 0),
        (255, 0, 0),
        (255, 100, 100),
    ],
    (255, 255, 0): [
        (128, 128, 0),
        (200, 200, 0),
        (255, 255, 0),
        (255, 255, 120),
    ],
    (0, 128, 255): [
        (0, 64, 128),
        (0, 100, 200),
        (0, 128, 255),
        (100, 180, 255),
    ],
    (0, 200, 0): [
        (0, 100, 0),
        (0, 150, 0),
        (0, 200, 0),
        (120, 255, 120),
    ],
}

BASE_COLORS: tuple[Color, ...] = tuple(COLOR_GRADIENTS.keys())


def get_gradient_for_base(base_color: Color) -> list[Color]:
    """Return the visual gradient for a base color."""
    return COLOR_GRADIENTS.get(base_color, [base_color])


def get_gradient_base_color(particle: object) -> Color | None:
    """Return the stable clear-color identity for a particle."""
    return getattr(particle, "base_color", getattr(particle, "color", None))


def get_color_index(color: Color | None) -> int | None:
    """Return the canonical color index for a base or gradient color."""
    if color is None:
        return None

    for index, (base, gradient) in enumerate(COLOR_GRADIENTS.items()):
        if color == base or color in gradient:
            return index

    return None
