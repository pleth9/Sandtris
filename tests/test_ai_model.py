import torch

from sandtris.ai.tetris_nn import TetrisNet


def test_tetris_net_preserves_spatial_path_to_actions():
    net = TetrisNet(input_channels=4, height=20, width=10, n_actions=9)
    inputs = torch.zeros((2, 4, 20, 10), dtype=torch.float32)

    outputs = net(inputs)

    assert outputs.shape == (2, 9)
    assert isinstance(net.flatten, torch.nn.Flatten)


def test_tetris_net_accepts_piece_metadata_path():
    net = TetrisNet(input_channels=4, height=20, width=10, n_actions=6, meta_features=28)
    board = torch.zeros((2, 4, 20, 10), dtype=torch.float32)
    metadata = torch.zeros((2, 28), dtype=torch.float32)

    outputs = net(board, metadata)

    assert outputs.shape == (2, 6)
