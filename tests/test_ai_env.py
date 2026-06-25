from sandtris.ai.actions import PlacementAction
from sandtris.ai.env import HeadlessSandtrisConfig, HeadlessSandtrisEnv


def make_env() -> HeadlessSandtrisEnv:
    return HeadlessSandtrisEnv(
        config=HeadlessSandtrisConfig(
            playfield_width=120,
            playfield_height=180,
            cell_size=6,
            box_size=18,
            action_interval=1,
            max_placement_frames=80,
        ),
        seed=1,
    )


def test_reset_returns_valid_observation():
    env = make_env()
    observation = env.reset(seed=2)

    assert observation.board.shape == (30, 20)
    assert observation.active_mask.shape == observation.board.shape
    assert observation.active_piece_features.shape == (5,)
    assert observation.next_piece_features.shape == (2,)


def test_step_returns_gym_like_contract_with_reward_debug_info():
    env = make_env()

    observation, reward, done, info = env.step("none")

    assert observation.board.shape == (30, 20)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "reward_components" in info
    assert "total_reward" in info["reward_components"]


def test_step_accepts_placement_action_scaffold():
    env = make_env()

    observation, reward, done, info = env.step(PlacementAction(rotation=0, target_column=5))

    assert observation.board.shape == (30, 20)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["action"] == "placement"
    assert info["placement_target_column"] == 5
    assert info["placement_frames"] > 0
