import numpy as np

from sandtris.ai.rewards import calculate_reward


def test_reward_returns_large_negative_on_game_over():
    board = np.zeros((6, 8), dtype=np.uint8)

    reward = calculate_reward(board, clear_points=0, game_over=True)

    assert reward.total_reward <= -50.0
    assert reward.game_over_penalty < 0


def test_reward_exposes_positive_clear_component():
    board = np.zeros((6, 8), dtype=np.uint8)

    reward = calculate_reward(board, clear_points=200, game_over=False, previous_bridge_potential=0.0)

    assert reward.clear_reward > 0
    assert reward.total_reward > 0
    assert reward.as_dict()["clear_reward"] == reward.clear_reward


def test_reward_penalizes_danger_cells_in_ghost_band():
    board = np.zeros((6, 8), dtype=np.uint8)
    board[0, 0] = 1
    board[1, 1] = 1

    reward = calculate_reward(board, clear_points=0, game_over=False, previous_bridge_potential=0.0)

    assert reward.danger_penalty < 0
    assert reward.total_reward < 0
