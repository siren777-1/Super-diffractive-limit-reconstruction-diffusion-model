import torch
import torch.nn as nn

class SCAU(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SCAU, self).__init__()
        r = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(r, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        energy = torch.mean(x ** 2, dim=(2, 3))  # [B, C]

        weights = self.fc(energy)  # [B, C]
        weights = weights.view(B, C, 1, 1)

        out = x * weights
        return out
