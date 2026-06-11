from sandtris.core import ScoreState, calculate_clear_score, get_level


def test_clear_score_rewards_size_level_combo_and_soft_drop():
    base = calculate_clear_score(200, level=1, combo=0)
    richer = calculate_clear_score(200, level=3, combo=2, soft_drop_points=5)

    assert richer > base


def test_score_state_tracks_level_combo_and_totals():
    state = ScoreState(score=4990)
    state.add_soft_drop(10)
    points = state.record_clear(200)

    assert points > 200
    assert state.score > 5000
    assert state.level == get_level(state.score)
    assert state.combo == 1
    assert state.pixels_cleared_total == 200


def test_no_clear_resets_combo_and_pending_soft_drop():
    state = ScoreState(combo=3, soft_drop_points_pending=4)

    state.record_no_clear()

    assert state.combo == 0
    assert state.soft_drop_points_pending == 0
