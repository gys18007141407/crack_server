import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet"]

class DoubleConv(nn.Module):
    """ (convolution => [BN] => ReLU) * 2 """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ Down sampling """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样为2倍大小
        x1 = self.up(x1)

        # input = [B, C, H, W]
        diff_2 = x2.size()[2] - x1.size()[2]
        diff_1 = x2.size()[3] - x1.size()[3]

        # 将x1填充为x2的形状
        x1 = F.pad(x1, [diff_1 // 2, diff_1 - diff_1 // 2,
                        diff_2 // 2, diff_2 - diff_2 // 2])

        # 跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 卷积增加channel提取原始信息
        self.inc = DoubleConv(n_channels, 32)
        # 下采样，编码提取低维信息
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        # 上采样，解码恢复原始信息
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        # 分类
        self.outc = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits