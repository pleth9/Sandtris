import csv
import json

import torch

from sandtris.ai.actions import N_CONTROL_ACTIONS
from sandtris.ai.env import HeadlessSandtrisConfig, HeadlessSandtrisEnv
from sandtris.ai.eval import run_benchmark
from sandtris.ai.observation import DEFAULT_ENCODER
from sandtris.ai.tetris_nn import TetrisNet


def small_config() -> HeadlessSandtrisConfig:
    return HeadlessSandtrisConfig(
        playfield_width=120,
        playfield_height=180,
        cell_size=6,
        box_size=18,
        action_interval=1,
        max_placement_frames=80,
    )


def test_random_eval_is_reproducible_for_same_seed():
    kwargs = {
        "policies": ["random"],
        "episodes": 2,
        "config": small_config(),
        "max_steps_per_episode": 5,
        "seed": 9,
        "device": "cpu",
    }

    first = run_benchmark(**kwargs)
    second = run_benchmark(**kwargs)

    assert first["results"][0]["episodes"] == second["results"][0]["episodes"]
    assert first["results"][0]["aggregate"] == second["results"][0]["aggregate"]


def test_benchmark_writes_json_and_csv(tmp_path):
    out_json = tmp_path / "ai_eval_results.json"
    out_csv = tmp_path / "ai_benchmark.csv"

    result = run_benchmark(
        policies=["random"],
        episodes=1,
        config=small_config(),
        max_steps_per_episode=3,
        seed=3,
        device="cpu",
        out_json=out_json,
        out_csv=out_csv,
    )

    payload = json.loads(out_json.read_text())
    assert payload["results"][0]["policy"] == "random"
    assert payload["results"][0]["aggregate"]["episodes"] == 1
    assert result["outputs"]["json"] == str(out_json)

    with out_csv.open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["policy"] == "random"
    assert rows[0]["seed"] == "3"
    assert "score" in rows[0]


def test_model_eval_loads_checkpoint(tmp_path):
    config = small_config()
    env = HeadlessSandtrisEnv(config=config, seed=0)
    rows, cols = env.simulation.grid.rows, env.simulation.grid.columns
    model = TetrisNet(
        input_channels=DEFAULT_ENCODER.input_channels,
        height=rows,
        width=cols,
        n_actions=N_CONTROL_ACTIONS,
        meta_features=DEFAULT_ENCODER.meta_features,
    )
    model_path = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), model_path)

    result = run_benchmark(
        policies=["model"],
        episodes=1,
        config=config,
        max_steps_per_episode=2,
        seed=4,
        device="cpu",
        model_path=model_path,
    )

    policy_result = result["results"][0]
    assert policy_result["policy"] == "model"
    assert policy_result["model_path"] == str(model_path)
    assert policy_result["aggregate"]["episodes"] == 1
