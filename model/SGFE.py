import torch
import torch.nn as nn

class SGFE(nn.Module):
    def __init__(self, in_channels, num_features=32):
        super(SGFE, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)

        self.tau = nn.Parameter(torch.zeros(1))  
        self.gamma = nn.Parameter(torch.ones(1))  
        self.relu = nn.ReLU(inplace=True)

        # OPTIONAL
        self.conv_out = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        Z1 = self.relu(self.conv1(x))
        Z2 = self.conv2(Z1)  # [B, C, H, W]


        energy = torch.sum(Z2 ** 2, dim=1, keepdim=True)  # [B, 1, H, W]


        G = torch.sigmoid(self.gamma * (energy - self.tau))  # [B, 1, H, W]


        out = Z2 * G

        out = self.conv_out(out)
        return out
