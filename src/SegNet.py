import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)

        # Decoder
        self.dec3 = self._block(256, 128)
        self.dec2 = self._block(128, 64)
        self.dec1 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, padding=1))

    def _block(self, in_channels, out_channels):
        """Defines a Conv -> BatchNorm -> ReLU block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x, ind1 = F.max_pool2d(
            self.enc1(x), kernel_size=2, stride=2, return_indices=True
        )
        x, ind2 = F.max_pool2d(
            self.enc2(x), kernel_size=2, stride=2, return_indices=True
        )
        x, ind3 = F.max_pool2d(
            self.enc3(x), kernel_size=2, stride=2, return_indices=True
        )

        # Decoder
        x = F.max_unpool2d(x, ind3, kernel_size=2, stride=2)
        x = self.dec3(x)
        x = F.max_unpool2d(x, ind2, kernel_size=2, stride=2)
        x = self.dec2(x)
        x = F.max_unpool2d(x, ind1, kernel_size=2, stride=2)
        x = self.dec1(x)

        return x
