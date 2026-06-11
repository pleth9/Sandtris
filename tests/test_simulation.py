from sandtris.core import Grid, SandParticle, SandTetromino, Simulation


RED = (255, 0, 0)


def test_sand_falls_down_when_empty():
    sim = Simulation(width=5, height=5, cell_size=1)
    sim.grid.set_cell(1, 2, SandParticle(color=RED, base_color=RED))

    sim.update()

    assert sim.grid.get_cell(1, 2) is None
    assert sim.grid.get_cell(2, 2) is not None


def test_sand_respects_floor_and_walls():
    sim = Simulation(width=3, height=3, cell_size=1)
    particle = SandParticle(color=RED, base_color=RED)
    sim.grid.set_cell(2, 0, particle)

    sim.update()

    assert sim.grid.get_cell(2, 0) is particle


def test_sand_uses_available_diagonal_when_blocked_below():
    grid = Grid(width=3, height=3, cell_size=1)
    particle = SandParticle(color=RED, base_color=RED)
    grid.set_cell(0, 1, particle)
    grid.set_cell(1, 1, SandParticle(color=RED, base_color=RED))

    assert particle.update(grid, 0, 1, frame=0) == (1, 2)
    assert particle.update(grid, 0, 1, frame=1) == (1, 0)


def test_sand_stays_put_when_down_and_diagonals_are_blocked():
    grid = Grid(width=3, height=3, cell_size=1)
    particle = SandParticle(color=RED, base_color=RED)
    grid.set_cell(0, 1, particle)
    grid.set_cell(1, 0, SandParticle(color=RED, base_color=RED))
    grid.set_cell(1, 1, SandParticle(color=RED, base_color=RED))
    grid.set_cell(1, 2, SandParticle(color=RED, base_color=RED))

    assert particle.update(grid, 0, 1, frame=0) == (0, 1)


def test_tetromino_breaks_on_ground_and_releases_sand():
    sim = Simulation(width=60, height=60, cell_size=3)
    tetromino = SandTetromino(0, 0, 12, shape="O", color=RED, base_color=RED)
    tetromino.y = 48
    tetromino.sync_boxes()

    tetromino.update(sim, ground_y=59, wall_left=0, wall_right=60)

    assert tetromino.broken
    assert tetromino.last_released_positions
    assert any(sim.grid.get_cell(row, col) is not None for row in range(sim.grid.rows) for col in range(sim.grid.columns))
