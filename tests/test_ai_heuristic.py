import numpy as np

from sandtris.ai.heuristic import (
    choose_heuristic_placement,
    enumerate_placement_actions,
    evaluate_placement_action,
    estimate_wall_bridge_potential,
    evaluate_board,
    iter_color_components,
    remove_wall_to_wall_bridges,
)
from sandtris.ai.env import HeadlessSandtrisConfig, HeadlessSandtrisEnv
from sandtris.core.constants import BASE_COLORS
from sandtris.core.tetromino import SandTetromino


def test_bridge_potential_rewards_connected_horizontal_span():
    tiny_scattered = np.zeros((6, 8), dtype=np.uint8)
    tiny_scattered[3, [0, 2, 4]] = 1
    connected_span = np.zeros((6, 8), dtype=np.uint8)
    connected_span[3, 0:5] = 1

    assert estimate_wall_bridge_potential(connected_span) > estimate_wall_bridge_potential(tiny_scattered)


def test_bridge_potential_strongly_rewards_actual_wall_bridge():
    partial = np.zeros((6, 8), dtype=np.uint8)
    partial[3, 0:5] = 1
    bridge = np.zeros((6, 8), dtype=np.uint8)
    bridge[3, :] = 1

    assert estimate_wall_bridge_potential(bridge) > estimate_wall_bridge_potential(partial)


def test_component_analysis_is_four_connected_by_color():
    board = np.zeros((4, 5), dtype=np.uint8)
    board[1, 0:2] = 1
    board[1, 4] = 1
    board[2, 4] = 2

    components = iter_color_components(board)

    assert len(components) == 3
    assert sorted(component.size for component in components) == [1, 1, 2]


def test_board_evaluation_penalizes_top_danger_cells():
    calm = np.zeros((6, 8), dtype=np.uint8)
    danger = calm.copy()
    danger[0, 3] = 1

    assert evaluate_board(danger).score < evaluate_board(calm).score


def test_placement_enumeration_exposes_rotation_and_target_columns():
    piece = SandTetromino(0, 0, 18, shape="T", color=BASE_COLORS[0], base_color=BASE_COLORS[0])

    actions = enumerate_placement_actions(piece, board_width=120, cell_size=6, box_size=18)

    assert actions
    assert {action.rotation for action in actions} == {0, 1, 2, 3}
    assert min(action.target_column for action in actions) >= 0


def test_wall_to_wall_bridge_removal_reports_clear_metrics():
    board = np.zeros((4, 5), dtype=np.uint8)
    board[2, :] = 1

    clear_pixels, clear_events, cleared = remove_wall_to_wall_bridges(board)

    assert clear_pixels == 5
    assert clear_events == 1
    assert not np.any(cleared[2, :])


def test_heuristic_policy_is_deterministic_for_fixed_state():
    config = HeadlessSandtrisConfig(
        playfield_width=120,
        playfield_height=180,
        cell_size=6,
        box_size=18,
        action_interval=1,
        max_placement_frames=80,
    )
    env_a = HeadlessSandtrisEnv(config=config, seed=4)
    env_b = HeadlessSandtrisEnv(config=config, seed=4)
    env_a.reset(seed=4)
    env_b.reset(seed=4)

    action_a = choose_heuristic_placement(env_a)
    action_b = choose_heuristic_placement(env_b)
    evaluation = evaluate_placement_action(env_a, action_a)

    assert action_a == action_b
    assert evaluation.action == action_a
    assert isinstance(evaluation.score, float)
