from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import math
import random
import time
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandtris.ai.tetris_nn import TetrisNet
from sandtris.core.constants import BASE_COLORS
from sandtris.core.scoring import calculate_clear_score, get_gravity, get_level
from sandtris.core.simulation import Simulation
from sandtris.core.tetromino import SandTetromino

CELL_SIZE = 3
PLAYFIELD_WIDTH = 492
PLAYFIELD_HEIGHT = 780
WALL_THICKNESS = 0
GHOST_ROWS = 2
BOX_SIZE = 36
ACTION_INTERVAL = 3
MOVE_SPEED = 10

ACTIONS = [
    "left",
    "right",
    "rotate_right",
    "rotate_left",
    "drop",
    "none",
    "hold_left",
    "hold_right",
    "hold_drop",
]
N_ACTIONS = len(ACTIONS)

MODEL_PATH = PROJECT_ROOT / "tetris_brain.pth"
STEPS_PATH = PROJECT_ROOT / "steps_done.json"
HEADLESS_STATS_PATH = PROJECT_ROOT / "headless_stats.json"

REPLAY_SIZE = 15000
BATCH_SIZE = 32
GAMMA = 0.995
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 7_500_000
LEARNING_RATE = 1e-5
TARGET_UPDATE_STEPS = 1000
TRAIN_EVERY_ACTIONS = 32
LEARNING_STARTS = 1000


def get_centered_spawn_x(tetromino_class: type[SandTetromino], playfield_width: int, box_size: int, **kwargs: object) -> int:
    temp = tetromino_class(0, 0, box_size, **kwargs)
    min_x = min(box.x for box in temp.boxes)
    max_x = max(box.x + box.size for box in temp.boxes)
    return int((playfield_width - (max_x - min_x)) // 2 - min_x)


def build_color_index(grid: object) -> np.ndarray:
    cached = grid.get_color_grid()
    return np.where(cached >= 0, cached + 1, 0).astype(np.uint8, copy=False)


def prepare_state(grid: object) -> torch.Tensor:
    """Return compact uint8 board IDs shaped as one batch item."""
    return torch.from_numpy(build_color_index(grid).copy()).unsqueeze(0)


def states_to_model_input(states: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Expand compact board IDs to the 4-channel float tensor the CNN expects."""
    states = states.to(device=device, non_blocking=True)
    return torch.stack([states == color_id for color_id in range(1, 5)], dim=1).to(dtype=torch.float32)


def get_column_heights(color_index: np.ndarray) -> np.ndarray:
    rows, cols = color_index.shape
    heights = np.zeros(cols, dtype=np.int16)
    present = color_index > 0
    top_indices = np.argmax(present, axis=0)
    has_sand = np.any(present, axis=0)
    heights[has_sand] = rows - top_indices[has_sand]
    return heights


def calculate_reward(color_index: np.ndarray, clear_points: float, game_over: bool) -> float:
    if game_over:
        return -1000.0

    heights = get_column_heights(color_index)
    max_height = int(np.max(heights)) if len(heights) else 0
    mean_height = float(np.mean(heights)) if len(heights) else 0.0
    height_ratio = max_height / max(1, color_index.shape[0])

    reward = -1.0
    reward += float(clear_points) * 0.5
    reward -= 0.02 * mean_height
    if height_ratio > 0.70:
        reward -= 200.0 * ((height_ratio - 0.70) / 0.30) ** 2
    return reward


class HeadlessSandtrisEnv:
    def __init__(self, cell_size: int, box_size: int, action_interval: int):
        self.cell_size = cell_size
        self.box_size = box_size
        self.action_interval = action_interval
        self.ground_y = PLAYFIELD_HEIGHT - 1
        self.spawn_y = -GHOST_ROWS * box_size
        self.move_left = False
        self.move_right = False
        self.speed_down = False
        self.move_hold_counter = 0
        self.soft_drop_frame_counter = 0
        self.reset()

    def reset(self) -> torch.Tensor:
        self.simulation = Simulation(PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT, self.cell_size)
        self.score = 0.0
        self.level = 1
        self.pixels_cleared_total = 0
        self.move_left = False
        self.move_right = False
        self.speed_down = False
        self.move_hold_counter = 0
        self.soft_drop_frame_counter = 0

        active_color = random.choice(BASE_COLORS)
        next_color = random.choice(BASE_COLORS)
        spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, self.box_size, color=active_color, base_color=active_color)
        next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, self.box_size, color=next_color, base_color=next_color)
        self.active_tetromino = SandTetromino(spawn_x, self.spawn_y, self.box_size, color=active_color, base_color=active_color)
        self.next_piece = SandTetromino(next_spawn_x, self.spawn_y, self.box_size, color=next_color, base_color=next_color)
        self.tetrominoes = [self.active_tetromino]
        self.game_over = False
        return prepare_state(self.simulation.grid)

    def step(self, action_idx: int) -> tuple[torch.Tensor, float, bool, dict[str, float]]:
        clear_points = 0.0
        action = ACTIONS[action_idx]
        self._apply_action(action)

        for _ in range(self.action_interval):
            clear_points += self._advance_frame()
            if self.game_over:
                break

        color_index = build_color_index(self.simulation.grid)
        reward = calculate_reward(color_index, clear_points, self.game_over)
        info = {
            "score": self.score,
            "level": self.level,
            "pixels_cleared": self.pixels_cleared_total,
            "clear_points": clear_points,
        }
        return prepare_state(self.simulation.grid), reward, self.game_over, info

    def _apply_action(self, action: str) -> None:
        base_gravity = get_gravity(self.level)
        can_rotate = all((box.y // self.cell_size) >= GHOST_ROWS for box in self.active_tetromino.boxes)

        if action == "hold_left":
            self.move_left = True
            self.move_right = False
            self.speed_down = False
        elif action == "hold_right":
            self.move_right = True
            self.move_left = False
            self.speed_down = False
        elif action == "hold_drop":
            self.speed_down = True
        elif action == "left":
            self.move_left = self.move_right = self.speed_down = False
            self.active_tetromino.x -= MOVE_SPEED
        elif action == "right":
            self.move_left = self.move_right = self.speed_down = False
            self.active_tetromino.x += MOVE_SPEED
        elif action == "rotate_right" and can_rotate:
            self.active_tetromino.rotate("right", wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS, grid=self.simulation.grid)
        elif action == "rotate_left" and can_rotate:
            self.active_tetromino.rotate("left", wall_left=WALL_THICKNESS, wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS, grid=self.simulation.grid)
        elif action == "drop":
            self.move_left = self.move_right = False
            self.speed_down = True
            self.active_tetromino.gravity = base_gravity * 2.5
        elif action == "none":
            self.move_left = self.move_right = self.speed_down = False

        self._clamp_active_piece()

    def _advance_frame(self) -> float:
        self._apply_held_movement()

        base_gravity = get_gravity(self.level)
        self.active_tetromino.gravity = base_gravity * 2.0 if self.speed_down else base_gravity

        for tetromino in self.tetrominoes:
            if not tetromino.broken:
                tetromino.update(
                    self.simulation,
                    self.ground_y,
                    wall_left=WALL_THICKNESS,
                    wall_right=PLAYFIELD_WIDTH - WALL_THICKNESS,
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
            self.active_tetromino.x -= MOVE_SPEED
        if self.move_right:
            self.active_tetromino.x += MOVE_SPEED
        self.move_hold_counter = 0
        self._clamp_active_piece()

    def _clamp_active_piece(self) -> None:
        min_dx = min(dx for dx, _dy in self.active_tetromino.offsets)
        max_dx = max(dx for dx, _dy in self.active_tetromino.offsets)
        left = self.active_tetromino.x + min_dx * self.box_size
        right = self.active_tetromino.x + (max_dx + 1) * self.box_size
        if left < WALL_THICKNESS:
            self.active_tetromino.x = WALL_THICKNESS - min_dx * self.box_size
        if right > PLAYFIELD_WIDTH - WALL_THICKNESS:
            self.active_tetromino.x = PLAYFIELD_WIDTH - WALL_THICKNESS - (max_dx + 1) * self.box_size
        self.active_tetromino.sync_boxes()

    def _resolve_clears(self) -> float:
        total_pixels = 0
        while True:
            cleared, _regions = self.simulation.grid.clear_wall_to_wall_regions()
            if cleared == 0:
                break
            total_pixels += cleared
            self.simulation.activate_positions({position for region in _regions for position in region.positions})

        if total_pixels == 0:
            return 0.0

        points = calculate_clear_score(total_pixels, level=self.level)
        self.score += points
        self.pixels_cleared_total += total_pixels
        self.level = get_level(self.score)
        return float(points)

    def _spawn_next_piece_if_needed(self) -> None:
        if not self.active_tetromino.broken:
            return

        grid = self.simulation.grid
        if any(grid.cells[row][col] is not None for row in range(GHOST_ROWS) for col in range(grid.columns)):
            self.game_over = True
            return

        self.active_tetromino = self.next_piece
        next_color = random.choice(BASE_COLORS)
        next_spawn_x = get_centered_spawn_x(SandTetromino, PLAYFIELD_WIDTH, self.box_size, color=next_color, base_color=next_color)
        self.next_piece = SandTetromino(next_spawn_x, self.spawn_y, self.box_size, color=next_color, base_color=next_color)
        self.tetrominoes.append(self.active_tetromino)
        self.active_tetromino.gravity = get_gravity(self.level)


def choose_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preferred)


def training_paths(cell_size: int, box_size: int) -> tuple[Path, Path, Path]:
    if cell_size == CELL_SIZE and box_size == BOX_SIZE:
        return MODEL_PATH, HEADLESS_STATS_PATH, STEPS_PATH

    suffix = f"cs{cell_size}_bs{box_size}"
    return (
        PROJECT_ROOT / f"tetris_brain_{suffix}.pth",
        PROJECT_ROOT / f"headless_stats_{suffix}.json",
        PROJECT_ROOT / f"steps_done_{suffix}.json",
    )


def load_stats(headless_stats_path: Path, steps_path: Path) -> dict[str, float]:
    for path in (headless_stats_path, steps_path):
        if not path.exists():
            continue
        try:
            with path.open("r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            continue
    return {}


def save_stats(stats: dict[str, float], headless_stats_path: Path, steps_path: Path) -> None:
    with headless_stats_path.open("w") as file:
        json.dump(stats, file, indent=2)
    with steps_path.open("w") as file:
        json.dump(stats, file, indent=2)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    model_path, headless_stats_path, steps_path = training_paths(args.cell_size, args.box_size)
    env = HeadlessSandtrisEnv(args.cell_size, args.box_size, args.action_interval)
    rows, cols = env.simulation.grid.rows, env.simulation.grid.columns
    net = TetrisNet(input_channels=4, height=rows, width=cols, n_actions=N_ACTIONS).to(device, dtype=torch.float32)

    if model_path.exists():
        try:
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded model from {model_path}")
        except RuntimeError as exc:
            print(f"Existing model is incompatible with this architecture; starting fresh. ({exc})")

    target_net = copy.deepcopy(net).to(device, dtype=torch.float32)
    target_net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay_buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=args.replay_size)

    stats = load_stats(headless_stats_path, steps_path)
    steps_done = int(stats.get("steps_done", 0))
    train_steps_done = int(stats.get("train_steps_done", 0))
    episode = int(stats.get("generation", 1))
    best_score = float(stats.get("max_score", 0))
    cumulative_score = float(stats.get("cumulative_score", 0.0))
    started_at = time.time()
    last_log_at = started_at
    last_log_steps = steps_done

    print(f"Headless training on {device}. Press Ctrl+C to checkpoint and stop.")
    print(f"Board: {rows}x{cols} cells, cell_size={args.cell_size}, box_size={args.box_size}")
    print(
        f"Actions: {N_ACTIONS}, replay: {args.replay_size}, batch: {args.batch_size}, "
        f"train_every={args.train_every}, learning_starts={args.learning_starts}, model: {model_path.name}"
    )

    try:
        completed_this_run = 0
        while args.episodes == 0 or completed_this_run < args.episodes:
            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0

            while episode_steps < args.max_steps_per_episode:
                epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
                if random.random() < epsilon:
                    action_idx = random.randrange(N_ACTIONS)
                else:
                    with torch.no_grad(), nullcontext():
                        q_values = net(states_to_model_input(state, device))
                        action_idx = int(q_values.argmax().item())

                next_state, reward, done, info = env.step(action_idx)
                replay_buffer.append((state, action_idx, reward, next_state, done))
                state = next_state
                episode_reward += reward
                steps_done += 1
                episode_steps += 1

                if len(replay_buffer) >= max(args.batch_size, args.learning_starts) and steps_done % args.train_every == 0:
                    loss = dqn_train_step(
                        net,
                        target_net,
                        optimizer,
                        replay_buffer,
                        device,
                        args.batch_size,
                    )
                    train_steps_done += 1
                    if train_steps_done % TARGET_UPDATE_STEPS == 0:
                        target_net.load_state_dict(net.state_dict())
                        target_net.eval()
                    if math.isnan(loss) or math.isinf(loss):
                        print(f"Skipping invalid loss: {loss}")

                now = time.time()
                if now - last_log_at >= args.log_every:
                    steps_per_second = (steps_done - last_log_steps) / max(0.001, now - last_log_at)
                    print(
                        f"ep={episode} step={steps_done} eps={epsilon:.3f} "
                        f"score={int(info['score'])} best={int(best_score)} "
                        f"reward={episode_reward:.1f} replay={len(replay_buffer)} "
                        f"{steps_per_second:.1f} decisions/s"
                    )
                    last_log_at = now
                    last_log_steps = steps_done

                if steps_done % args.checkpoint_every == 0:
                    save_checkpoint(net, model_path, headless_stats_path, steps_path, steps_done, train_steps_done, episode, cumulative_score, best_score)

                if done:
                    break

            cumulative_score += env.score
            best_score = max(best_score, env.score)
            print(
                f"episode {episode} done: score={int(env.score)} "
                f"pixels={env.pixels_cleared_total} reward={episode_reward:.1f} steps={episode_steps}"
            )
            episode += 1
            completed_this_run += 1
            save_checkpoint(net, model_path, headless_stats_path, steps_path, steps_done, train_steps_done, episode, cumulative_score, best_score)
    except KeyboardInterrupt:
        print("\nStopping headless training after Ctrl+C.")
    finally:
        save_checkpoint(net, model_path, headless_stats_path, steps_path, steps_done, train_steps_done, episode, cumulative_score, best_score)


def dqn_train_step(
    net: TetrisNet,
    target_net: TetrisNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]],
    device: torch.device,
    batch_size: int,
) -> float:
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states_tensor = states_to_model_input(torch.cat(states), device)
    next_states_tensor = states_to_model_input(torch.cat(next_states), device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device)

    optimizer.zero_grad()
    q_values = net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = net(next_states_tensor).argmax(1)
        next_q = target_net(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards_tensor + GAMMA * next_q * (~dones_tensor)
    loss = F.smooth_l1_loss(q_values, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(
    net: TetrisNet,
    model_path: Path,
    headless_stats_path: Path,
    steps_path: Path,
    steps_done: int,
    train_steps_done: int,
    episode: int,
    cumulative_score: float,
    best_score: float,
) -> None:
    torch.save(net.state_dict(), model_path)
    save_stats(
        {
            "steps_done": steps_done,
            "train_steps_done": train_steps_done,
            "generation": episode,
            "cumulative_score": cumulative_score,
            "max_score": best_score,
            "start_time": time.time(),
        },
        headless_stats_path,
        steps_path,
    )
    print(f"checkpoint saved: steps={steps_done}, episode={episode}, best={int(best_score)}, model={model_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Sandtris DQN trainer.")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to train; 0 means forever.")
    parser.add_argument("--max-steps-per-episode", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=float, default=10.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--replay-size", type=int, default=REPLAY_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train-every", type=int, default=TRAIN_EVERY_ACTIONS)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS, help="Replay items to collect before the first optimizer step.")
    parser.add_argument("--cell-size", type=int, default=6, help="Headless training grid cell size. Larger is faster/coarser.")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE, help="Tetromino block size in pixels.")
    parser.add_argument("--action-interval", type=int, default=ACTION_INTERVAL, help="Simulation frames per DQN decision.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
