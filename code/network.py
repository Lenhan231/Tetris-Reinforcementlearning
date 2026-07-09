"""Deep Q-Network for Tetris game state evaluation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
class DeepQNetwork(nn.Module):
    """Deep Q-Network: maps game state features to Q-value estimate.

    Architecture: input_size → hidden1 → hidden2 → 1 (Q-value)
    Default: 4 → 64 → 64 → 1
    """
    def __init__(self, input_size=4, hidden1_size=64, hidden2_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass: state → Q-value."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def count_params(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    

