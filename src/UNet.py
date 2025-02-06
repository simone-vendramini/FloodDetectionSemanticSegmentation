import torch
import torchvision
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        torch.nn.init.xavier_normal_(self.depthwise.weight)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        torch.nn.init.xavier_normal_(self.pointwise.weight)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleConvBlockSeparable(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlockSeparable, self).__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1.bias, 0)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2.bias, 0)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, separable=True):
        super(DownSampleBlock, self).__init__()
        if separable:
            self.double_conv = DoubleConvBlockSeparable(in_channels, out_channels)
        else:
            self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.maxpool_conv = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        skip = x
        x = self.maxpool_conv(x)
        return x, skip


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, separable=True):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        torch.nn.init.xavier_normal_(self.up.weight)
        torch.nn.init.constant_(self.up.bias, 0)
        if separable:
            self.double_conv = DoubleConvBlockSeparable(in_channels, out_channels)
        else:
            self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        skip = torchvision.transforms.functional.center_crop(skip, x.shape[2:])

        x = torch.cat((x, skip), dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, separable=True):
        super(UNet, self).__init__()
        # Contracting part
        self.down1 = DownSampleBlock(in_channels, 16, separable)
        self.down2 = DownSampleBlock(16, 32, separable)
        self.down3 = DownSampleBlock(32, 64, separable)
        self.down4 = DownSampleBlock(64, 128, separable)

        # Bottleneck
        if separable:
            self.bottleneck = DoubleConvBlockSeparable(128, 256)
        else:
            self.bottleneck = DoubleConvBlock(128, 256)

        # Expansive part
        self.up1 = UpSampleBlock(256, 128, separable)
        self.up2 = UpSampleBlock(128, 64, separable)
        self.up3 = UpSampleBlock(64, 32, separable)
        self.up4 = UpSampleBlock(32, 16, separable)

        # Final layer
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.final_conv(x)
        return x
