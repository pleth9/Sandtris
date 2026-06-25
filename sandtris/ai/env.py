"""Pygame-free Sandtris environment for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random

from sandtris.ai.actions import CONTROL_ACTIONS, PlacementAction, normalize_control_action
from sandtris.ai.heuristic import estimate_wall_bridge_potential
from sandtris.ai.observation import SandtrisObservation, build_color_index, make_observation
from sandtris.ai.rewards import DEFAULT_REWARD_CONFIG, RewardBreakdown, RewardConfig, calculate_reward
from sandtris.core.constants import BASE_COLORS
from sandtris.core.scoring import ScoreState, get_gravity
from sandtris.core.simulation import Simulation
from sandtris.core.tetromino import SandTetromino, TETROMINO_SHAPES


@dataclass(frozen=True)
class HeadlessSandtrisConfig:
    playfield_width: int = 492
    playfield_height: int = 780
    wall_thickness: int = 0
    ghost_rows: int = 2
    cell_size: int = 6
    box_size: int = 36
    action_interval: int = 3
    move_speed: int = 10
    max_placement_frames: int = 1000
    placement_settle_frames: int = 0


DEFAULT_CONFIG = HeadlessSandtrisConfig()


class HeadlessSandtrisEnv:
    """Small gym-like environment: ``reset()`` and ``step(action)``."""

    control_actions = CONTROL_ACTIONS

    def __init__(
        self,
        config: HeadlessSandtrisConfig | None = None,
        *,
        seed: int | None = None,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    ):
        self.config = config if config is not None else DEFAULT_CONFIG
        self.reward_config = reward_config
        self.rng = random.Random(seed)
        self.ground_y = self.config.playfield_height - 1
        self.spawn_y = -self.config.ghost_rows * self.config.box_size
        self.reset()

    @property
    def score(self) -> int:
        return self.score_state.score

    @property
    def level(self) -> int:
        return self.score_state.level

    @property
    def pixels_cleared_total(self) -> int:
        return self.score_state.pixels_cleared_total

    def reset(self, *, seed: int | None = None) -> SandtrisObservation:
        if seed is not None:
            self.rng.seed(seed)

        self.simulation = Simulation(
            self.config.playfield_width,
            self.config.playfield_height,
            self.config.cell_size,
        )
        self.score_state = ScoreState()
        self.move_left = False
        self.move_right = False
        self.speed_down = False
        self.move_hold_counter = 0
        self.soft_drop_frame_counter = 0
        self.game_over = False

        self.active_tetromino = self._new_piece()
        self.next_piece = self._new_piece()
        self.tetrominoes = [self.active_tetromino]
        self._last_bridge_potential = estimate_wall_bridge_potential(build_color_index(self.simulation.grid))
        return self.observe()

    def observe(self) -> SandtrisObservation:
        return make_observation(
            self.simulation.grid,
            self.active_tetromino,
            self.next_piece,
            cell_size=self.config.cell_size,
            score=float(self.score),
            level=self.level,
            ghost_rows=self.config.ghost_rows,
            gravity=get_gravity(self.level),
        )

    def step(
        self,
        action: int | str | PlacementAction,
    ) -> tuple[SandtrisObservation, float, bool, dict[str, object]]:
        if isinstance(action, PlacementAction):
            return self.step_placement(action)

        action_name = normalize_control_action(action)
        self._apply_control_action(action_name)
        clear_points = 0.0
        for _ in range(self.config.action_interval):
            clear_points += self._advance_frame()
            if self.game_over:
                break

        return self._finish_step(action_name, clear_points)

    def step_placement(
        self,
        action: PlacementAction,
    ) -> tuple[SandtrisObservation, float, bool, dict[str, object]]:
        original_piece = self.active_tetromino
        self._apply_placement_action(action)
        clear_points = 0.0
        frames = 0
        while not self.game_over and self.active_tetromino is original_piece and frames < self.config.max_placement_frames:
            clear_points += self._advance_frame()
            frames += 1

        for _ in range(self.config.placement_settle_frames):
            if self.game_over:
                break
            clear_points += self._advance_frame()

        observation, reward, done, info = self._finish_step("placement", clear_points)
        info["placement_frames"] = frames
        info["placement_rotation"] = action.rotation % 4
        info["placement_target_column"] = action.target_column
        return observation, reward, done, info

    def _finish_step(
        self,
        action_name: str,
        clear_points: float,
    ) -> tuple[SandtrisObservation, float, bool, dict[str, object]]:
        board = build_color_index(self.simulation.grid)
        reward = calculate_reward(
            board,
            clear_points=clear_points,
            game_over=self.game_over,
            previous_bridge_potential=self._last_bridge_potential,
            config=self.reward_config,
        )
        self._last_bridge_potential = reward.bridge_potential
        info = self._info(action_name, clear_points, reward)
        return self.observe(), reward.total_reward, self.game_over, info

    def _info(self, action_name: str, clear_points: float, reward: RewardBreakdown) -> dict[str, object]:
        return {
            "action": action_name,
            "score": float(self.score),
            "level": self.level,
            "pixels_cleared": self.pixels_cleared_total,
            "clear_points": float(clear_points),
            "reward_components": reward.as_dict(),
        }

    def _apply_control_action(self, action: str) -> None:
        can_rotate = all((box.y // self.config.cell_size) >= self.config.ghost_rows for box in self.active_tetromino.boxes)

        if action == "left":
            self.move_left = self.move_right = self.speed_down = False
            self.active_tetromino.x -= self.config.move_speed
        elif action == "right":
            self.move_left = self.move_right = self.speed_down = False
            self.active_tetromino.x += self.config.move_speed
        elif action == "rotate_right" and can_rotate:
            self.active_tetromino.rotate(
                "right",
                wall_left=self.config.wall_thickness,
                wall_right=self.config.playfield_width - self.config.wall_thickness,
                grid=self.simulation.grid,
            )
        elif action == "rotate_left" and can_rotate:
            self.active_tetromino.rotate(
                "left",
                wall_left=self.config.wall_thickness,
                wall_right=self.config.playfield_width - self.config.wall_thickness,
                grid=self.simulation.grid,
            )
        elif action == "drop":
            self.move_left = self.move_right = False
            self.speed_down = True
        elif action == "none":
            self.move_left = self.move_right = self.speed_down = False

        self._clamp_active_piece()

    def _apply_placement_action(self, action: PlacementAction) -> None:
        target_rotation = action.rotation % 4
        attempts = 0
        while self.active_tetromino.rotation % 4 != target_rotation and attempts < 4:
            if not self.active_tetromino.rotate(
                "right",
                wall_left=self.config.wall_thickness,
                wall_right=self.config.playfield_width - self.config.wall_thickness,
                grid=self.simulation.grid,
            ):
                break
            attempts += 1
        min_dx = min(dx for dx, _dy in self.active_tetromino.offsets)
        self.active_tetromino.x = action.target_column * self.config.cell_size - min_dx * self.config.box_size
        self.active_tetromino.sync_boxes()
        self._clamp_active_piece()
        self.move_left = self.move_right = False
        self.speed_down = True

    def _advance_frame(self) -> float:
        self._apply_held_movement()
        self._apply_gravity()

        for tetromino in self.tetrominoes:
            if not tetromino.broken:
                tetromino.update(
                    self.simulation,
                    self.ground_y,
                    wall_left=self.config.wall_thickness,
                    wall_right=self.config.playfield_width - self.config.wall_thickness,
                )

        self.simulation.update()
        clear_points = self._resolve_clears()
        self._spawn_next_piece_if_needed()
        return clear_points

    def _apply_held_movement(self) -> None:
        self.move_hold_counter += 1
        if self.move_hold_counter < 1:
            return
        if self.move_left:
            self.active_tetromino.x -= self.config.move_speed
        if self.move_right:
            self.active_tetromino.x += self.config.move_speed
        self.move_hold_counter = 0
        self._clamp_active_piece()

    def _apply_gravity(self) -> None:
        base_gravity = get_gravity(self.level)
        if self.speed_down:
            self.active_tetromino.gravity = base_gravity * 2.5
            self.soft_drop_frame_counter += 1
            if self.soft_drop_frame_counter % 2 == 0:
                self.score_state.add_soft_drop(1)
        else:
            self.active_tetromino.gravity = base_gravity
            self.soft_drop_frame_counter = 0

    def _clamp_active_piece(self) -> None:
        min_dx = min(dx for dx, _dy in self.active_tetromino.offsets)
        max_dx = max(dx for dx, _dy in self.active_tetromino.offsets)
        left = self.active_tetromino.x + min_dx * self.config.box_size
        right = self.active_tetromino.x + (max_dx + 1) * self.config.box_size
        wall_left = self.config.wall_thickness
        wall_right = self.config.playfield_width - self.config.wall_thickness
        if left < wall_left:
            self.active_tetromino.x = wall_left - min_dx * self.config.box_size
        if right > wall_right:
            self.active_tetromino.x = wall_right - (max_dx + 1) * self.config.box_size
        self.active_tetromino.sync_boxes()

    def _resolve_clears(self) -> float:
        total_cleared = 0
        while True:
            cleared, regions = self.simulation.grid.clear_wall_to_wall_regions()
            if cleared == 0:
                break
            total_cleared += cleared
            self.simulation.activate_positions({position for region in regions for position in region.positions})

        if not total_cleared:
            return 0.0
        return float(self.score_state.record_clear(total_cleared))

    def _spawn_next_piece_if_needed(self) -> None:
        if not self.active_tetromino.broken:
            return

        if self.has_sand_in_ghost_rows():
            self.game_over = True
            return

        self.active_tetromino = self.next_piece
        self.next_piece = self._new_piece()
        self.tetrominoes.append(self.active_tetromino)
        self.active_tetromino.gravity = get_gravity(self.level)

    def has_sand_in_ghost_rows(self) -> bool:
        grid = self.simulation.grid
        return any(grid.cells[row][col] is not None for row in range(self.config.ghost_rows) for col in range(grid.columns))

    def _new_piece(self) -> SandTetromino:
        color = self.rng.choice(BASE_COLORS)
        shape = self.rng.choice(tuple(TETROMINO_SHAPES))
        spawn_x = get_centered_spawn_x(
            SandTetromino,
            self.config.playfield_width,
            self.config.box_size,
            shape=shape,
            color=color,
            base_color=color,
        )
        return SandTetromino(
            spawn_x,
            self.spawn_y,
            self.config.box_size,
            shape=shape,
            color=color,
            base_color=color,
        )


def get_centered_spawn_x(
    tetromino_class: type[SandTetromino],
    playfield_width: int,
    box_size: int,
    **kwargs: object,
) -> int:
    """Calculate a spawn X that centers the piece after SRS offsets."""
    temp = tetromino_class(0, 0, box_size, **kwargs)
    min_x = min(box.x for box in temp.boxes)
    max_x = max(box.x + box.size for box in temp.boxes)
    return int((playfield_width - (max_x - min_x)) // 2 - min_x)


__all__ = [
    "DEFAULT_CONFIG",
    "HeadlessSandtrisConfig",
    "HeadlessSandtrisEnv",
    "get_centered_spawn_x",
]
