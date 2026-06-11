from sandtris.core import Grid, find_wall_to_wall_clears


RED = (255, 0, 0)
BLUE = (0, 128, 255)


def make_grid(rows=6, cols=8):
    return Grid(width=cols, height=rows, cell_size=1)


def fill(grid, positions, color=RED):
    for row, col in positions:
        assert grid.add_particle_with_color(row, col, color, base_color=color)


def test_wall_to_wall_connected_color_clear_succeeds():
    grid = make_grid()
    fill(grid, [(2, col) for col in range(grid.columns)])

    regions = find_wall_to_wall_clears(grid)

    assert len(regions) == 1
    assert regions[0].size == grid.columns
    cleared, _ = grid.clear_wall_to_wall_regions()
    assert cleared == grid.columns
    assert all(grid.get_cell(2, col) is None for col in range(grid.columns))


def test_disconnected_same_color_does_not_clear():
    grid = make_grid()
    fill(grid, [(2, 0), (2, 1), (2, 6), (2, 7)])

    assert find_wall_to_wall_clears(grid) == []
    cleared, _ = grid.clear_wall_to_wall_regions()
    assert cleared == 0


def test_different_color_bridge_does_not_clear():
    grid = make_grid()
    fill(grid, [(2, 0), (2, 1), (2, 2)], RED)
    fill(grid, [(2, 3)], BLUE)
    fill(grid, [(2, 4), (2, 5), (2, 6), (2, 7)], RED)

    assert find_wall_to_wall_clears(grid) == []


def test_non_straight_connected_path_clears():
    grid = make_grid(rows=5, cols=6)
    path = [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5)]
    fill(grid, path)

    regions = find_wall_to_wall_clears(grid)

    assert len(regions) == 1
    assert regions[0].positions == frozenset(path)


def test_multiple_simultaneous_regions_clear():
    grid = make_grid(rows=5, cols=5)
    fill(grid, [(1, col) for col in range(5)], RED)
    fill(grid, [(3, col) for col in range(5)], BLUE)

    cleared, regions = grid.clear_wall_to_wall_regions()

    assert cleared == 10
    assert len(regions) == 2
