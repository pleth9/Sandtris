import numpy as np
import torch

from sandtris.ai.env import HeadlessSandtrisConfig, HeadlessSandtrisEnv, get_centered_spawn_x
from sandtris.ai.observation import DEFAULT_ENCODER, SHAPE_TO_ID
from sandtris.core.constants import BASE_COLORS, get_color_index
from sandtris.core.tetromino import SandTetromino


RED = BASE_COLORS[0]
BLUE = BASE_COLORS[1]


def make_small_env() -> HeadlessSandtrisEnv:
    return HeadlessSandtrisEnv(
        config=HeadlessSandtrisConfig(
            playfield_width=120,
            playfield_height=180,
            cell_size=6,
            box_size=18,
            action_interval=1,
            max_placement_frames=80,
        ),
        seed=4,
    )


def set_known_pieces(env: HeadlessSandtrisEnv) -> None:
    active_x = get_centered_spawn_x(
        SandTetromino,
        env.config.playfield_width,
        env.config.box_size,
        shape="T",
        color=RED,
        base_color=RED,
    )
    next_x = get_centered_spawn_x(
        SandTetromino,
        env.config.playfield_width,
        env.config.box_size,
        shape="L",
        color=BLUE,
        base_color=BLUE,
    )
    env.active_tetromino = SandTetromino(
        active_x,
        env.config.ghost_rows * env.config.cell_size,
        env.config.box_size,
        shape="T",
        color=RED,
        base_color=RED,
    )
    env.next_piece = SandTetromino(
        next_x,
        env.spawn_y,
        env.config.box_size,
        shape="L",
        color=BLUE,
        base_color=BLUE,
    )
    env.tetrominoes = [env.active_tetromino]


def test_active_piece_movement_changes_observation_without_board_change():
    env = make_small_env()
    set_known_pieces(env)
    before = env.observe()

    env.active_tetromino.x += env.config.cell_size
    env.active_tetromino.sync_boxes()
    after = env.observe()

    assert np.array_equal(before.board, after.board)
    assert not np.array_equal(before.active_piece_features, after.active_piece_features)
    assert not np.array_equal(before.active_mask, after.active_mask)


def test_active_piece_rotation_changes_observation():
    env = make_small_env()
    set_known_pieces(env)
    before = env.observe()

    rotated = env.active_tetromino.rotate(
        "right",
        wall_left=env.config.wall_thickness,
        wall_right=env.config.playfield_width - env.config.wall_thickness,
        grid=env.simulation.grid,
    )
    after = env.observe()

    assert rotated
    assert before.active_piece_features[2] != after.active_piece_features[2]
    assert not np.array_equal(before.active_mask, after.active_mask)


def test_observation_includes_active_and_next_piece_identity():
    env = make_small_env()
    set_known_pieces(env)
    observation = env.observe()

    assert observation.active_piece_features[0] == SHAPE_TO_ID["T"]
    assert observation.active_piece_features[1] == get_color_index(RED) + 1
    assert observation.next_piece_features[0] == SHAPE_TO_ID["L"]
    assert observation.next_piece_features[1] == get_color_index(BLUE) + 1


def test_encoder_returns_board_overlay_channels_and_metadata():
    env = make_small_env()
    set_known_pieces(env)
    observation = env.observe()

    board_channels, metadata = DEFAULT_ENCODER.encode((observation,), device=torch.device("cpu"))

    assert board_channels.shape == (1, DEFAULT_ENCODER.input_channels, *observation.board.shape)
    assert board_channels.dtype == torch.float32
    assert metadata.shape == (1, DEFAULT_ENCODER.meta_features)
