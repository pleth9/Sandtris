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
        seed=12,
    )


def test_placement_action_accepts_target_x_alias():
    action = PlacementAction(rotation=2, target_x=7)

    assert action.target_column == 7
    assert action.target_x == 7


def test_placement_action_advances_piece_or_settles_sand():
    env = make_env()
    original_piece = env.active_tetromino

    observation, reward, done, info = env.step(PlacementAction(rotation=0, target_x=5))

    assert observation.board.shape == (30, 20)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["action"] == "placement"
    assert info["placement_target_x"] == 5
    assert info["placement_frames"] <= env.config.max_placement_frames
    assert env.active_tetromino is not original_piece or env.simulation.grid.particle_count > 0 or done


def test_impossible_placement_columns_are_clamped_without_crashing():
    env = make_env()

    observation, reward, done, info = env.step(PlacementAction(rotation=3, target_x=9999))

    assert observation.board.shape == (30, 20)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["action"] == "placement"
    assert info["placement_frames"] <= env.config.max_placement_frames
