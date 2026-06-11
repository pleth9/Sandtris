"""Sand particle movement."""

from __future__ import annotations

import colorsys
import random

from sandtris.core.constants import Color


class SandParticle:
    """Single settled or falling sand cell."""

    def __init__(self, color: Color | None = None, base_color: Color | None = None):
        self.color = color if color is not None else random_color((0.1, 0.12), (0.5, 0.7), (0.7, 0.9))
        self.base_color = base_color if base_color is not None else self.color

    def update(self, grid: object, row: int, column: int, frame: int = 0) -> tuple[int, int]:
        """Return the next cell using deterministic, alternating side checks."""
        if grid.is_cell_empty(row + 1, column):
            return row + 1, column

        parity = (row + column + frame) % 2
        offsets = (-1, 1) if parity == 0 else (1, -1)
        for offset in offsets:
            next_column = column + offset
            if grid.is_cell_empty(row + 1, next_column):
                return row + 1, next_column

        return row, column


def random_color(
    hue_range: tuple[float, float],
    saturation_range: tuple[float, float],
    value_range: tuple[float, float],
) -> Color:
    """Generate an RGB color from HSV ranges."""
    hue = random.uniform(*hue_range)
    saturation = random.uniform(*saturation_range)
    value = random.uniform(*value_range)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255)
