"""Compatibility wrapper for old imports."""

from sandtris.core.grid import Grid
from sandtris.core.constants import COLOR_GRADIENTS, get_color_index, get_gradient_base_color, get_gradient_for_base

__all__ = [
    "COLOR_GRADIENTS",
    "Grid",
    "get_color_index",
    "get_gradient_base_color",
    "get_gradient_for_base",
]
