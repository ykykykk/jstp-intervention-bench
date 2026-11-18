import torch
from torch import nn

class ConvGRU(nn.Module):
    def __init__(self, input_channels: int = 112, hidden_channels: int = 128):
        super().__init__()
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, prior: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((observation, prior), dim=1)).sigmoid().chunk(2, dim=1)
        proposal = self.candidate(torch.cat((observation, reset * prior), dim=1)).tanh()
        return (1 - update) * prior + update * proposal
