"""Action definitions for headless Sandtris agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ControlAction(IntEnum):
    """Frame-level controls matching the playable Sandtris inputs."""

    LEFT = 0
    RIGHT = 1
    ROTATE_RIGHT = 2
    ROTATE_LEFT = 3
    DROP = 4
    NONE = 5


CONTROL_ACTIONS: tuple[str, ...] = (
    "left",
    "right",
    "rotate_right",
    "rotate_left",
    "drop",
    "none",
)
N_CONTROL_ACTIONS = len(CONTROL_ACTIONS)


@dataclass(frozen=True)
class PlacementAction:
    """Placement-level action scaffold: choose orientation and target column."""

    rotation: int
    target_column: int


def normalize_control_action(action: int | str | ControlAction) -> str:
    """Return the canonical control action name."""
    if isinstance(action, ControlAction):
        return CONTROL_ACTIONS[int(action)]
    if isinstance(action, int):
        try:
            return CONTROL_ACTIONS[action]
        except IndexError as exc:
            raise ValueError(f"unknown control action index: {action}") from exc
    if action in CONTROL_ACTIONS:
        return action
    raise ValueError(f"unknown control action: {action!r}")


__all__ = [
    "CONTROL_ACTIONS",
    "N_CONTROL_ACTIONS",
    "ControlAction",
    "PlacementAction",
    "normalize_control_action",
]
