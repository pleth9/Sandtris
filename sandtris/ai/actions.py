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


@dataclass(frozen=True, init=False)
class PlacementAction:
    """Placement-level action: choose orientation and target grid column.

    ``target_column`` is the canonical name used by the headless environment.
    ``target_x`` is accepted as a compatibility alias because placement search
    code often describes the same value as a target x-column.
    """

    rotation: int
    target_column: int

    def __init__(
        self,
        rotation: int,
        target_column: int | None = None,
        *,
        target_x: int | None = None,
    ) -> None:
        if target_column is None and target_x is None:
            raise TypeError("PlacementAction requires target_column or target_x")
        if target_column is not None and target_x is not None and target_column != target_x:
            raise ValueError("target_column and target_x must match when both are provided")
        object.__setattr__(self, "rotation", int(rotation))
        object.__setattr__(self, "target_column", int(target_column if target_column is not None else target_x))

    @property
    def target_x(self) -> int:
        """Alias for ``target_column``."""
        return self.target_column


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
