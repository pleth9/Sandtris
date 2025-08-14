import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TetrisNet(nn.Module):
    """Memory-optimized CNN for Sandtris with LayerNorm and attention."""
    def __init__(self, input_channels: int, height: int, width: int, n_actions: int):
        """Initialize the neural network with specified input dimensions and action count."""
        super().__init__()
        self.n_actions = n_actions
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.ln1 = nn.LayerNorm([32, height, width])
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.ln2 = nn.LayerNorm([32, height, width])
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.ln3 = nn.LayerNorm([32, height, width])
        
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.fc1 = nn.Linear(32, 128, bias=False)
        self.fc2 = nn.Linear(128, n_actions)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights for better convergence."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.ln1(self.conv1(x)))
        
        identity = x
        out = F.relu(self.ln2(self.conv2(x)))
        out = self.ln3(self.conv3(out))
        x = F.relu(out + identity)
        
        att = self.attention(x)
        x = x * att
        
        x = x.mean(dim=(-2, -1))
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)