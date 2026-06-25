"""Headless DQN training entrypoint for Sandtris."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sandtris.ai.actions import CONTROL_ACTIONS, N_CONTROL_ACTIONS
from sandtris.ai.env import DEFAULT_CONFIG, HeadlessSandtrisConfig, HeadlessSandtrisEnv
from sandtris.ai.observation import DEFAULT_ENCODER, ObservationEncoder, SandtrisObservation
from sandtris.ai.tetris_nn import TetrisNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPLAY_SIZE = 15_000
BATCH_SIZE = 32
GAMMA = 0.995
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 7_500_000
LEARNING_RATE = 1e-5
TARGET_UPDATE_STEPS = 1000
TRAIN_EVERY_ACTIONS = 32
LEARNING_STARTS = 1000

METRIC_FIELDS = (
    "event",
    "wall_time",
    "step",
    "episode",
    "epsilon",
    "score",
    "episode_reward",
    "episode_steps",
    "pixels_cleared",
    "best_score",
    "eval_avg",
    "eval_median",
    "eval_best",
    "best_eval",
    "replay_size",
    "decisions_per_second",
    "train_steps",
    "loss",
    "loss_mean",
    "q_mean",
    "q_max",
    "learning_rate",
    "eval_reward",
    "eval_pixels",
    "eval_steps",
)


ReplayItem = tuple[SandtrisObservation, int, float, SandtrisObservation, bool]


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = HeadlessSandtrisConfig(
        cell_size=args.cell_size,
        box_size=args.box_size,
        action_interval=args.action_interval,
        max_placement_frames=args.max_placement_frames,
    )
    encoder = DEFAULT_ENCODER
    device = choose_device(args.device)
    model_path, stats_path, steps_path, metrics_path = training_paths(args.output_dir, config)
    if args.eval_only:
        run_eval_only(args, config, device, model_path)
        return

    env = HeadlessSandtrisEnv(config=config, seed=args.seed)
    rows, cols = env.simulation.grid.rows, env.simulation.grid.columns
    net = TetrisNet(
        input_channels=encoder.input_channels,
        height=rows,
        width=cols,
        n_actions=N_CONTROL_ACTIONS,
        meta_features=encoder.meta_features,
    ).to(device, dtype=torch.float32)

    if model_path.exists() and not args.fresh:
        try:
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded model from {model_path}")
        except RuntimeError as exc:
            print(f"Existing checkpoint is incompatible; starting fresh. ({exc})")

    target_net = copy.deepcopy(net).to(device, dtype=torch.float32)
    target_net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay_buffer: deque[ReplayItem] = deque(maxlen=args.replay_size)

    stats = {} if args.fresh else load_stats(stats_path, steps_path)
    steps_done = int(stats.get("steps_done", 0))
    train_steps_done = int(stats.get("train_steps_done", 0))
    episode = int(stats.get("episode", stats.get("generation", 1)))
    best_score = float(stats.get("best_score", stats.get("max_score", 0)))
    best_eval_score = float(stats.get("best_eval_score", 0))
    last_eval_score = float(stats.get("last_eval_score", 0))
    last_eval_median = float(stats.get("last_eval_median", 0))
    last_eval_steps = int(stats.get("last_eval_steps", 0))
    cumulative_score = float(stats.get("cumulative_score", 0.0))
    started_at = time.time()
    last_log_at = started_at
    last_log_steps = steps_done
    last_loss: float | None = None
    recent_losses: deque[float] = deque(maxlen=100)
    recent_q_means: deque[float] = deque(maxlen=100)
    recent_q_maxes: deque[float] = deque(maxlen=100)
    checkpoint_current = False

    print(f"Headless Sandtris training on {device}. Press Ctrl+C to checkpoint and stop.")
    print(f"Board: {rows}x{cols} cells, cell_size={config.cell_size}, box_size={config.box_size}")
    print(
        f"Observation: {encoder.input_channels} board channels + {encoder.meta_features} metadata features "
        "(settled board, active piece, next piece, score/level/gravity)."
    )
    print(
        f"Actions: {N_CONTROL_ACTIONS} ({', '.join(CONTROL_ACTIONS)}), replay={args.replay_size}, "
        f"batch={args.batch_size}, train_every={args.train_every}, learning_starts={args.learning_starts}"
    )
    print(f"Outputs: {model_path.name}, {stats_path.name}, {metrics_path.name}")

    try:
        completed_this_run = 0
        while args.episodes == 0 or completed_this_run < args.episodes:
            state = env.reset(seed=args.seed + episode if args.seed_each_episode else None)
            episode_reward = 0.0
            episode_steps = 0

            while episode_steps < args.max_steps_per_episode:
                epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
                action_idx, q_mean, q_max = choose_action_with_diagnostics(net, encoder, state, device, epsilon)
                if q_mean is not None:
                    recent_q_means.append(q_mean)
                if q_max is not None:
                    recent_q_maxes.append(q_max)
                next_state, reward, done, info = env.step(action_idx)
                replay_buffer.append((state, action_idx, reward, next_state, done))

                state = next_state
                episode_reward += reward
                steps_done += 1
                checkpoint_current = False
                episode_steps += 1

                if len(replay_buffer) >= max(args.batch_size, args.learning_starts) and steps_done % args.train_every == 0:
                    last_loss = dqn_train_step(net, target_net, optimizer, replay_buffer, encoder, device, args.batch_size)
                    train_steps_done += 1
                    if train_steps_done % TARGET_UPDATE_STEPS == 0:
                        target_net.load_state_dict(net.state_dict())
                        target_net.eval()
                    if math.isnan(last_loss) or math.isinf(last_loss):
                        print(f"Skipping invalid loss value: {last_loss}")
                    else:
                        recent_losses.append(last_loss)

                now = time.time()
                if now - last_log_at >= args.log_every:
                    decisions_per_second = (steps_done - last_log_steps) / max(0.001, now - last_log_at)
                    loss_mean = mean_recent(recent_losses)
                    q_mean_recent = mean_recent(recent_q_means)
                    q_max_recent = mean_recent(recent_q_maxes)
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    loss_text = "" if loss_mean is None else f" loss={loss_mean:.4f}"
                    q_text = "" if q_mean_recent is None else f" q_mean={q_mean_recent:.3f} q_max={q_max_recent:.3f}"
                    print(
                        f"ep={episode} step={steps_done} eps={epsilon:.3f} "
                        f"score={int(float(info['score']))} pixels={int(info['pixels_cleared'])} "
                        f"reward={episode_reward:.1f} replay={len(replay_buffer)} "
                        f"lr={learning_rate:.2g}{loss_text}{q_text} "
                        f"{decisions_per_second:.1f} decisions/s"
                    )
                    append_metric(
                        metrics_path,
                        event="log",
                        wall_time=int(now),
                        step=steps_done,
                        episode=episode,
                        epsilon=f"{epsilon:.6f}",
                        score=f"{float(info['score']):.3f}",
                        episode_reward=f"{episode_reward:.3f}",
                        episode_steps=episode_steps,
                        pixels_cleared=int(info["pixels_cleared"]),
                        best_score=f"{best_score:.3f}",
                        best_eval=f"{best_eval_score:.3f}",
                        replay_size=len(replay_buffer),
                        decisions_per_second=f"{decisions_per_second:.3f}",
                        train_steps=train_steps_done,
                        loss="" if last_loss is None else f"{last_loss:.6f}",
                        loss_mean="" if loss_mean is None else f"{loss_mean:.6f}",
                        q_mean="" if q_mean_recent is None else f"{q_mean_recent:.6f}",
                        q_max="" if q_max_recent is None else f"{q_max_recent:.6f}",
                        learning_rate=f"{learning_rate:.10f}",
                    )
                    last_log_at = now
                    last_log_steps = steps_done

                if args.checkpoint_every > 0 and steps_done % args.checkpoint_every == 0:
                    save_checkpoint(
                        net,
                        model_path,
                        stats_path,
                        steps_path,
                        steps_done,
                        train_steps_done,
                        episode,
                        cumulative_score,
                        best_score,
                        best_eval_score,
                        last_eval_score,
                        last_eval_median,
                        last_eval_steps,
                    )
                    checkpoint_current = True

                if args.eval_every > 0 and steps_done % args.eval_every == 0 and last_eval_steps != steps_done:
                    eval_stats = evaluate_policy(
                        net,
                        encoder,
                        device,
                        config,
                        args.eval_episodes,
                        args.max_steps_per_episode,
                        args.eval_epsilon,
                        args.seed + 100_000,
                    )
                    last_eval_score = eval_stats["avg_score"]
                    last_eval_median = eval_stats["median_score"]
                    best_eval_score = max(best_eval_score, last_eval_score)
                    last_eval_steps = steps_done
                    loss_mean = mean_recent(recent_losses)
                    q_mean_recent = mean_recent(recent_q_means)
                    q_max_recent = mean_recent(recent_q_maxes)
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    print(
                        f"eval step={steps_done} eps={args.eval_epsilon:.3f} "
                        f"avg={last_eval_score:.1f} median={last_eval_median:.1f} "
                        f"reward={eval_stats['mean_total_reward']:.2f} pixels={eval_stats['mean_pixels_cleared']:.1f} "
                        f"best_eval={best_eval_score:.1f}"
                    )
                    append_metric(
                        metrics_path,
                        event="eval",
                        wall_time=int(time.time()),
                        step=steps_done,
                        episode=episode,
                        epsilon=f"{args.eval_epsilon:.6f}",
                        best_score=f"{best_score:.3f}",
                        eval_avg=f"{last_eval_score:.3f}",
                        eval_median=f"{last_eval_median:.3f}",
                        eval_best=f"{eval_stats['best_score']:.3f}",
                        best_eval=f"{best_eval_score:.3f}",
                        replay_size=len(replay_buffer),
                        train_steps=train_steps_done,
                        loss="" if last_loss is None else f"{last_loss:.6f}",
                        loss_mean="" if loss_mean is None else f"{loss_mean:.6f}",
                        q_mean="" if q_mean_recent is None else f"{q_mean_recent:.6f}",
                        q_max="" if q_max_recent is None else f"{q_max_recent:.6f}",
                        learning_rate=f"{learning_rate:.10f}",
                        eval_reward=f"{eval_stats['mean_total_reward']:.6f}",
                        eval_pixels=f"{eval_stats['mean_pixels_cleared']:.6f}",
                        eval_steps=f"{eval_stats['mean_episode_length']:.6f}",
                    )

                if done:
                    break

            cumulative_score += env.score
            best_score = max(best_score, float(env.score))
            print(
                f"episode {episode} done: score={int(env.score)} pixels={env.pixels_cleared_total} "
                f"reward={episode_reward:.1f} steps={episode_steps}"
            )
            loss_mean = mean_recent(recent_losses)
            q_mean_recent = mean_recent(recent_q_means)
            q_max_recent = mean_recent(recent_q_maxes)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            append_metric(
                metrics_path,
                event="episode",
                wall_time=int(time.time()),
                step=steps_done,
                episode=episode,
                epsilon=f"{max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY):.6f}",
                score=f"{float(env.score):.3f}",
                episode_reward=f"{episode_reward:.3f}",
                episode_steps=episode_steps,
                pixels_cleared=env.pixels_cleared_total,
                best_score=f"{best_score:.3f}",
                best_eval=f"{best_eval_score:.3f}",
                replay_size=len(replay_buffer),
                train_steps=train_steps_done,
                loss="" if last_loss is None else f"{last_loss:.6f}",
                loss_mean="" if loss_mean is None else f"{loss_mean:.6f}",
                q_mean="" if q_mean_recent is None else f"{q_mean_recent:.6f}",
                q_max="" if q_max_recent is None else f"{q_max_recent:.6f}",
                learning_rate=f"{learning_rate:.10f}",
            )
            episode += 1
            completed_this_run += 1
            save_checkpoint(
                net,
                model_path,
                stats_path,
                steps_path,
                steps_done,
                train_steps_done,
                episode,
                cumulative_score,
                best_score,
                best_eval_score,
                last_eval_score,
                last_eval_median,
                last_eval_steps,
            )
            checkpoint_current = True
    except KeyboardInterrupt:
        print("\nStopping headless training after Ctrl+C.")
    finally:
        if not checkpoint_current:
            save_checkpoint(
                net,
                model_path,
                stats_path,
                steps_path,
                steps_done,
                train_steps_done,
                episode,
                cumulative_score,
                best_score,
                best_eval_score,
                last_eval_score,
                last_eval_median,
                last_eval_steps,
            )


def choose_action(
    net: TetrisNet,
    encoder: ObservationEncoder,
    state: SandtrisObservation,
    device: torch.device,
    epsilon: float,
) -> int:
    action, _q_mean, _q_max = choose_action_with_diagnostics(net, encoder, state, device, epsilon)
    return action


def choose_action_with_diagnostics(
    net: TetrisNet,
    encoder: ObservationEncoder,
    state: SandtrisObservation,
    device: torch.device,
    epsilon: float,
) -> tuple[int, float | None, float | None]:
    if random.random() < epsilon:
        return random.randrange(N_CONTROL_ACTIONS), None, None
    with torch.no_grad():
        board_input, meta_input = encoder.encode((state,), device)
        q_values = net(board_input, meta_input)
        return int(q_values.argmax().item()), float(q_values.mean().item()), float(q_values.max().item())


def evaluate_policy(
    net: TetrisNet,
    encoder: ObservationEncoder,
    device: torch.device,
    config: HeadlessSandtrisConfig,
    episodes: int,
    max_steps_per_episode: int,
    epsilon: float,
    seed: int,
) -> dict[str, float]:
    if episodes <= 0:
        return {
            "avg_score": 0.0,
            "median_score": 0.0,
            "best_score": 0.0,
            "mean_total_reward": 0.0,
            "mean_pixels_cleared": 0.0,
            "mean_episode_length": 0.0,
        }

    from sandtris.ai.eval import evaluate_model_policy

    result = evaluate_model_policy(
        net,
        encoder,
        device,
        config,
        episodes,
        max_steps_per_episode,
        epsilon,
        seed,
    )
    aggregate = result["aggregate"]
    return {
        "avg_score": float(aggregate["mean_score"]),
        "median_score": float(aggregate["median_score"]),
        "best_score": float(aggregate["max_score"]),
        "mean_total_reward": float(aggregate["mean_total_reward"]),
        "mean_pixels_cleared": float(aggregate["mean_pixels_cleared"]),
        "mean_episode_length": float(aggregate["mean_episode_length"]),
    }


def run_eval_only(
    args: argparse.Namespace,
    config: HeadlessSandtrisConfig,
    device: torch.device,
    default_model_path: Path,
) -> None:
    from sandtris.ai.eval import print_benchmark_summary, run_benchmark

    episodes = args.episodes if args.episodes > 0 else args.eval_episodes
    model_path: str | Path | None = args.model_path
    if args.policy == "model" and model_path is None:
        if default_model_path.exists():
            model_path = default_model_path
        else:
            print("No model checkpoint provided or found; evaluating fresh model weights.")

    out_json, out_csv = resolve_eval_output_paths(args)
    benchmark = run_benchmark(
        policies=[args.policy],
        episodes=episodes,
        config=config,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        device=device,
        model_path=model_path,
        epsilon=args.eval_epsilon,
        random_action_mode=args.random_action_mode,
        out_json=out_json,
        out_csv=out_csv,
    )
    print_benchmark_summary(benchmark)


def dqn_train_step(
    net: TetrisNet,
    target_net: TetrisNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: deque[ReplayItem],
    encoder: ObservationEncoder,
    device: torch.device,
    batch_size: int,
) -> float:
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states_board, states_meta = encoder.encode(states, device)
    next_states_board, next_states_meta = encoder.encode(next_states, device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device)

    optimizer.zero_grad()
    q_values = net(states_board, states_meta).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = net(next_states_board, next_states_meta).argmax(1)
        next_q = target_net(next_states_board, next_states_meta).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards_tensor + GAMMA * next_q * (~dones_tensor)
    loss = F.smooth_l1_loss(q_values, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def choose_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def training_paths(
    output_dir: str | Path,
    config: HeadlessSandtrisConfig,
) -> tuple[Path, Path, Path, Path]:
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    suffix = f"cs{config.cell_size}_bs{config.box_size}"
    model_path = directory / f"tetris_brain_observable_v1_{suffix}.pth"
    stats_path = directory / f"headless_stats_observable_v1_{suffix}.json"
    steps_path = directory / f"steps_done_observable_v1_{suffix}.json"
    metrics_path = directory / f"training_metrics_observable_v1_{suffix}.csv"
    return model_path, stats_path, steps_path, metrics_path


def load_stats(stats_path: Path, steps_path: Path) -> dict[str, float]:
    for path in (stats_path, steps_path):
        if not path.exists():
            continue
        try:
            with path.open("r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            continue
    return {}


def save_checkpoint(
    net: TetrisNet,
    model_path: Path,
    stats_path: Path,
    steps_path: Path,
    steps_done: int,
    train_steps_done: int,
    episode: int,
    cumulative_score: float,
    best_score: float,
    best_eval_score: float,
    last_eval_score: float,
    last_eval_median: float,
    last_eval_steps: int,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_path)
    stats = {
        "steps_done": steps_done,
        "train_steps_done": train_steps_done,
        "episode": episode,
        "cumulative_score": cumulative_score,
        "best_score": best_score,
        "best_eval_score": best_eval_score,
        "last_eval_score": last_eval_score,
        "last_eval_median": last_eval_median,
        "last_eval_steps": last_eval_steps,
        "updated_at": time.time(),
    }
    for path in (stats_path, steps_path):
        with path.open("w") as file:
            json.dump(stats, file, indent=2)
    print(
        f"checkpoint saved: steps={steps_done}, episode={episode}, "
        f"best={int(best_score)}, best_eval={best_eval_score:.1f}, model={model_path.name}"
    )


def append_metric(metrics_path: Path, **values: object) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metrics_path.exists()
    with metrics_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: values.get(field, "") for field in METRIC_FIELDS})


def resolve_eval_output_paths(args: argparse.Namespace) -> tuple[str | None, str | None]:
    out_json = args.eval_out_json
    out_csv = args.eval_out_csv
    if args.eval_out:
        suffix = Path(args.eval_out).suffix.lower()
        if suffix == ".csv":
            out_csv = args.eval_out
        else:
            out_json = args.eval_out
    return out_json, out_csv


def mean_recent(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Sandtris DQN trainer.")
    parser.add_argument("--episodes", type=int, default=0, help="Episodes to train; 0 means forever.")
    parser.add_argument("--max-steps-per-episode", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=float, default=10.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-each-episode", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint/stat files.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--replay-size", type=int, default=REPLAY_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train-every", type=int, default=TRAIN_EVERY_ACTIONS)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--eval-every", type=int, default=50_000, help="Decisions between eval runs; 0 disables eval.")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation without training or checkpoint writes.")
    parser.add_argument("--policy", choices=("model", "random", "heuristic"), default="model", help="Policy for --eval-only.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path for --eval-only model evaluation.")
    parser.add_argument("--eval-out", default=None, help="Evaluation output path; .csv writes CSV, otherwise JSON.")
    parser.add_argument("--eval-out-json", default=None)
    parser.add_argument("--eval-out-csv", default=None)
    parser.add_argument("--random-action-mode", choices=("control", "placement"), default="control")
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CONFIG.cell_size)
    parser.add_argument("--box-size", type=int, default=DEFAULT_CONFIG.box_size)
    parser.add_argument("--action-interval", type=int, default=DEFAULT_CONFIG.action_interval)
    parser.add_argument("--max-placement-frames", type=int, default=DEFAULT_CONFIG.max_placement_frames)
    parser.add_argument("--output-dir", default=".", help="Directory for checkpoints, stats, and metrics.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
