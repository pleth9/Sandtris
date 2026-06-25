"""Evaluation and benchmarking for headless Sandtris AI policies."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from sandtris.ai.actions import N_CONTROL_ACTIONS, PlacementAction
from sandtris.ai.env import DEFAULT_CONFIG, HeadlessSandtrisConfig, HeadlessSandtrisEnv
from sandtris.ai.heuristic import choose_heuristic_placement, enumerate_placement_actions
from sandtris.ai.observation import DEFAULT_ENCODER, ObservationEncoder, SandtrisObservation
from sandtris.ai.tetris_nn import TetrisNet

POLICY_CHOICES = ("random", "heuristic", "model")
CSV_FIELDS = (
    "policy",
    "seed",
    "episode",
    "score",
    "pixels_cleared",
    "total_reward",
    "steps",
    "frames",
    "clear_events",
    "done_reason",
    "model_path",
    "device",
)


@dataclass(frozen=True)
class EpisodeMetrics:
    policy: str
    seed: int
    episode: int
    score: float
    pixels_cleared: int
    total_reward: float
    steps: int
    frames: int
    clear_events: int
    done_reason: str
    q_mean: float | None = None
    q_max: float | None = None


class RandomPolicy:
    name = "random"
    model_path: str | None = None
    device: str | None = None

    def __init__(self, action_mode: str = "control"):
        if action_mode not in {"control", "placement"}:
            raise ValueError(f"unknown random action mode: {action_mode}")
        self.action_mode = action_mode
        self.rng = random.Random()

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def choose_action(self, env: HeadlessSandtrisEnv, observation: SandtrisObservation, step: int) -> int | PlacementAction:
        del observation, step
        if self.action_mode == "placement":
            actions = enumerate_placement_actions(
                env.active_tetromino,
                board_width=env.config.playfield_width,
                cell_size=env.config.cell_size,
                box_size=env.config.box_size,
                wall_left=env.config.wall_thickness,
                wall_right=env.config.playfield_width - env.config.wall_thickness,
            )
            return self.rng.choice(actions) if actions else PlacementAction(rotation=0, target_column=0)
        return self.rng.randrange(N_CONTROL_ACTIONS)

    def diagnostics(self) -> dict[str, float | None]:
        return {"q_mean": None, "q_max": None}


class HeuristicPolicy:
    name = "heuristic"
    model_path: str | None = None
    device: str | None = None

    def reset(self, seed: int) -> None:
        del seed

    def choose_action(self, env: HeadlessSandtrisEnv, observation: SandtrisObservation, step: int) -> PlacementAction:
        del observation, step
        return choose_heuristic_placement(env)

    def diagnostics(self) -> dict[str, float | None]:
        return {"q_mean": None, "q_max": None}


class ModelPolicy:
    name = "model"

    def __init__(
        self,
        net: TetrisNet,
        encoder: ObservationEncoder,
        device: torch.device,
        *,
        epsilon: float = 0.0,
        model_path: str | None = None,
    ):
        self.net = net
        self.encoder = encoder
        self.device = str(device)
        self._device = device
        self.epsilon = epsilon
        self.model_path = model_path
        self.rng = random.Random()
        self._q_mean: float | None = None
        self._q_max: float | None = None

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)
        self._q_mean = None
        self._q_max = None

    def choose_action(self, env: HeadlessSandtrisEnv, observation: SandtrisObservation, step: int) -> int:
        del env, step
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            self._q_mean = None
            self._q_max = None
            return self.rng.randrange(N_CONTROL_ACTIONS)

        with torch.no_grad():
            board_input, meta_input = self.encoder.encode((observation,), self._device)
            q_values = self.net(board_input, meta_input)
            self._q_mean = float(q_values.mean().item())
            self._q_max = float(q_values.max().item())
            return int(q_values.argmax().item())

    def diagnostics(self) -> dict[str, float | None]:
        return {"q_mean": self._q_mean, "q_max": self._q_max}


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible evaluation runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def make_model_policy(
    *,
    config: HeadlessSandtrisConfig,
    device: torch.device,
    encoder: ObservationEncoder = DEFAULT_ENCODER,
    epsilon: float = 0.0,
    model_path: str | Path | None = None,
) -> ModelPolicy:
    probe_env = HeadlessSandtrisEnv(config=config, seed=0)
    rows, cols = probe_env.simulation.grid.rows, probe_env.simulation.grid.columns
    net = TetrisNet(
        input_channels=encoder.input_channels,
        height=rows,
        width=cols,
        n_actions=N_CONTROL_ACTIONS,
        meta_features=encoder.meta_features,
    ).to(device, dtype=torch.float32)

    resolved_model_path: str | None = None
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        net.load_state_dict(checkpoint)
        resolved_model_path = str(path)

    net.eval()
    return ModelPolicy(net, encoder, device, epsilon=epsilon, model_path=resolved_model_path)


def run_policy_episodes(
    policy: RandomPolicy | HeuristicPolicy | ModelPolicy,
    *,
    config: HeadlessSandtrisConfig,
    episodes: int,
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive for evaluation")

    set_global_seed(seed)
    env = HeadlessSandtrisEnv(config=config, seed=seed)
    episode_metrics: list[EpisodeMetrics] = []

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        set_global_seed(episode_seed)
        policy.reset(episode_seed)
        observation = env.reset(seed=episode_seed)
        total_reward = 0.0
        steps = 0
        frames = 0
        clear_events = 0
        previous_pixels = env.pixels_cleared_total
        q_means: list[float] = []
        q_maxes: list[float] = []
        done = False

        for step_index in range(max_steps_per_episode):
            action = policy.choose_action(env, observation, step_index)
            observation, reward, done, info = env.step(action)
            steps += 1
            total_reward += reward
            frames += int(info.get("placement_frames", config.action_interval))

            pixels = int(info["pixels_cleared"])
            if pixels > previous_pixels:
                clear_events += 1
            previous_pixels = pixels

            diagnostics = policy.diagnostics()
            if diagnostics["q_mean"] is not None:
                q_means.append(float(diagnostics["q_mean"]))
            if diagnostics["q_max"] is not None:
                q_maxes.append(float(diagnostics["q_max"]))

            if done:
                break

        done_reason = "game_over" if done else "max_steps"
        episode_metrics.append(
            EpisodeMetrics(
                policy=policy.name,
                seed=episode_seed,
                episode=episode_index,
                score=float(env.score),
                pixels_cleared=int(env.pixels_cleared_total),
                total_reward=float(total_reward),
                steps=int(steps),
                frames=int(frames),
                clear_events=int(clear_events),
                done_reason=done_reason,
                q_mean=_mean_or_none(q_means),
                q_max=_mean_or_none(q_maxes),
            )
        )

    return {
        "policy": policy.name,
        "model_path": policy.model_path,
        "device": policy.device,
        "aggregate": aggregate_episode_metrics(episode_metrics),
        "episodes": [asdict(metrics) for metrics in episode_metrics],
    }


def evaluate_model_policy(
    net: TetrisNet,
    encoder: ObservationEncoder,
    device: torch.device,
    config: HeadlessSandtrisConfig,
    episodes: int,
    max_steps_per_episode: int,
    epsilon: float,
    seed: int,
) -> dict[str, object]:
    was_training = net.training
    net.eval()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        policy = ModelPolicy(net, encoder, device, epsilon=epsilon)
        return run_policy_episodes(
            policy,
            config=config,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if was_training:
            net.train()


def aggregate_episode_metrics(episodes: Sequence[EpisodeMetrics]) -> dict[str, float | int | None]:
    scores = [metrics.score for metrics in episodes]
    pixels = [float(metrics.pixels_cleared) for metrics in episodes]
    rewards = [metrics.total_reward for metrics in episodes]
    steps = [float(metrics.steps) for metrics in episodes]
    frames = [float(metrics.frames) for metrics in episodes]
    clear_events = [float(metrics.clear_events) for metrics in episodes]
    q_means = [metrics.q_mean for metrics in episodes if metrics.q_mean is not None]
    q_maxes = [metrics.q_max for metrics in episodes if metrics.q_max is not None]
    return {
        "episodes": len(episodes),
        "mean_score": _mean(scores),
        "median_score": float(statistics.median(scores)),
        "min_score": float(min(scores)),
        "max_score": float(max(scores)),
        "std_score": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        "mean_pixels_cleared": _mean(pixels),
        "mean_total_reward": _mean(rewards),
        "mean_episode_length": _mean(steps),
        "mean_frames": _mean(frames),
        "mean_clear_events": _mean(clear_events),
        "total_clear_events": int(sum(clear_events)),
        "mean_q_mean": _mean_or_none(q_means),
        "mean_q_max": _mean_or_none(q_maxes),
    }


def run_benchmark(
    *,
    policies: Sequence[str],
    episodes: int,
    config: HeadlessSandtrisConfig | None = None,
    max_steps_per_episode: int = 10_000,
    seed: int = 0,
    device: str | torch.device = "cpu",
    model_path: str | Path | None = None,
    epsilon: float = 0.0,
    random_action_mode: str = "control",
    out_json: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> dict[str, object]:
    config = DEFAULT_CONFIG if config is None else config
    torch_device = choose_device(str(device)) if not isinstance(device, torch.device) else device
    selected_policies = tuple(dict.fromkeys(policies))
    if not selected_policies:
        raise ValueError("at least one policy is required")

    outputs = {
        "json": None if out_json is None else str(out_json),
        "csv": None if out_csv is None else str(out_csv),
    }
    benchmark: dict[str, object] = {
        "generated_at": int(time.time()),
        "config": {
            "seed": seed,
            "episodes": episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "device": str(torch_device),
            "epsilon": epsilon,
            "random_action_mode": random_action_mode,
            "model_path": None if model_path is None else str(model_path),
            "environment": asdict(config),
        },
        "outputs": outputs,
        "results": [],
    }

    results: list[dict[str, object]] = []
    for policy_name in selected_policies:
        if policy_name not in POLICY_CHOICES:
            raise ValueError(f"unknown policy: {policy_name}")
        set_global_seed(seed)
        policy = build_policy(
            policy_name,
            config=config,
            device=torch_device,
            model_path=model_path,
            epsilon=epsilon,
            random_action_mode=random_action_mode,
        )
        results.append(
            run_policy_episodes(
                policy,
                config=config,
                episodes=episodes,
                max_steps_per_episode=max_steps_per_episode,
                seed=seed,
            )
        )

    benchmark["results"] = results
    if out_json is not None:
        write_json_results(Path(out_json), benchmark)
    if out_csv is not None:
        write_csv_results(Path(out_csv), benchmark)
    return benchmark


def build_policy(
    policy_name: str,
    *,
    config: HeadlessSandtrisConfig,
    device: torch.device,
    model_path: str | Path | None,
    epsilon: float,
    random_action_mode: str,
) -> RandomPolicy | HeuristicPolicy | ModelPolicy:
    if policy_name == "random":
        policy = RandomPolicy(action_mode=random_action_mode)
        policy.device = str(device)
        return policy
    if policy_name == "heuristic":
        policy = HeuristicPolicy()
        policy.device = str(device)
        return policy
    if policy_name == "model":
        return make_model_policy(config=config, device=device, epsilon=epsilon, model_path=model_path)
    raise ValueError(f"unknown policy: {policy_name}")


def write_json_results(path: Path, benchmark: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(benchmark, file, indent=2)


def write_csv_results(path: Path, benchmark: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in benchmark["results"]:
            model_path = result.get("model_path") if isinstance(result, dict) else None
            device = result.get("device") if isinstance(result, dict) else None
            episodes = result.get("episodes", []) if isinstance(result, dict) else []
            for episode in episodes:
                row = {field: episode.get(field, "") for field in CSV_FIELDS}
                row["model_path"] = model_path or ""
                row["device"] = device or ""
                writer.writerow(row)


def print_benchmark_summary(benchmark: dict[str, object]) -> None:
    for result in benchmark["results"]:
        aggregate = result["aggregate"]
        print(
            f"{result['policy']}: episodes={aggregate['episodes']} "
            f"mean_score={aggregate['mean_score']:.1f} median={aggregate['median_score']:.1f} "
            f"min={aggregate['min_score']:.1f} max={aggregate['max_score']:.1f} "
            f"std={aggregate['std_score']:.1f} pixels={aggregate['mean_pixels_cleared']:.1f} "
            f"reward={aggregate['mean_total_reward']:.2f} steps={aggregate['mean_episode_length']:.1f} "
            f"clears={aggregate['mean_clear_events']:.2f}"
        )
    outputs = benchmark.get("outputs", {})
    if isinstance(outputs, dict):
        if outputs.get("json"):
            print(f"wrote JSON: {outputs['json']}")
        if outputs.get("csv"):
            print(f"wrote CSV: {outputs['csv']}")


def config_from_args(args: argparse.Namespace) -> HeadlessSandtrisConfig:
    return HeadlessSandtrisConfig(
        playfield_width=args.playfield_width,
        playfield_height=args.playfield_height,
        wall_thickness=args.wall_thickness,
        ghost_rows=args.ghost_rows,
        cell_size=args.cell_size,
        box_size=args.box_size,
        action_interval=args.action_interval,
        max_placement_frames=args.max_placement_frames,
        placement_settle_frames=args.placement_settle_frames,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sandtris AI policies headlessly.")
    parser.add_argument("--policy", choices=POLICY_CHOICES, default=None, help="Single policy to evaluate.")
    parser.add_argument("--policies", nargs="+", choices=POLICY_CHOICES, default=None, help="Policies to benchmark.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration epsilon for model eval.")
    parser.add_argument("--random-action-mode", choices=("control", "placement"), default="control")
    parser.add_argument("--out", default=None, help="Output path; .csv writes CSV, everything else writes JSON.")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--playfield-width", type=int, default=DEFAULT_CONFIG.playfield_width)
    parser.add_argument("--playfield-height", type=int, default=DEFAULT_CONFIG.playfield_height)
    parser.add_argument("--wall-thickness", type=int, default=DEFAULT_CONFIG.wall_thickness)
    parser.add_argument("--ghost-rows", type=int, default=DEFAULT_CONFIG.ghost_rows)
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CONFIG.cell_size)
    parser.add_argument("--box-size", type=int, default=DEFAULT_CONFIG.box_size)
    parser.add_argument("--action-interval", type=int, default=DEFAULT_CONFIG.action_interval)
    parser.add_argument("--max-placement-frames", type=int, default=DEFAULT_CONFIG.max_placement_frames)
    parser.add_argument("--placement-settle-frames", type=int, default=DEFAULT_CONFIG.placement_settle_frames)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_json, out_csv = resolve_output_paths(args)
    policies = args.policies if args.policies is not None else [args.policy or "model"]
    benchmark = run_benchmark(
        policies=policies,
        episodes=args.episodes,
        config=config_from_args(args),
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        device=args.device,
        model_path=args.model_path,
        epsilon=args.epsilon,
        random_action_mode=args.random_action_mode,
        out_json=out_json,
        out_csv=out_csv,
    )
    print_benchmark_summary(benchmark)


def resolve_output_paths(args: argparse.Namespace) -> tuple[str | None, str | None]:
    out_json = args.out_json
    out_csv = args.out_csv
    if args.out:
        suffix = Path(args.out).suffix.lower()
        if suffix == ".csv":
            out_csv = args.out
        else:
            out_json = args.out
    return out_json, out_csv


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else _mean(values)


if __name__ == "__main__":
    main()
