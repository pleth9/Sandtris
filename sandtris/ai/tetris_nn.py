"""CNN model used by the optional Sandtris DQN trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TetrisNet(nn.Module):
    """Spatial CNN for Sandtris DQN.

    The network downsamples with strided convolutions, then flattens the
    resulting feature map. That keeps coarse board position information without
    the huge fully connected layer that a raw full-resolution flatten would
    require.
    """

    def __init__(self, input_channels: int, height: int, width: int, n_actions: int, meta_features: int = 0):
        super().__init__()
        self.n_actions = n_actions
        self.meta_features = meta_features

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(8, 32)
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        reduced_height = self._stride2_size(self._stride2_size(self._stride2_size(height)))
        reduced_width = self._stride2_size(self._stride2_size(self._stride2_size(width)))
        conv_features = 32 * reduced_height * reduced_width
        if meta_features:
            self.board_features = nn.Linear(conv_features, 256)
            self.meta_net = nn.Sequential(
                nn.Linear(meta_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            self.fc1 = nn.Linear(320, 256)
        else:
            self.board_features = None
            self.meta_net = None
            self.fc1 = nn.Linear(conv_features, 256)
        self.fc2 = nn.Linear(256, n_actions)
        self.apply(self._init_weights)

    @staticmethod
    def _stride2_size(size: int) -> int:
        return (size + 1) // 2

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = x * self.attention(x)
        x = self.flatten(x)
        if self.meta_net is not None:
            if meta is None:
                raise ValueError("metadata tensor is required when meta_features is non-zero")
            if self.board_features is None:
                raise ValueError("board feature projection is required when meta_features is non-zero")
            x = F.relu(self.board_features(x))
            x = torch.cat((x, self.meta_net(meta)), dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
