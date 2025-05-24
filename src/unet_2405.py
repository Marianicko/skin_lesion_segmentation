from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int | None = None,
            dropout_prob: float = 0.2,  # Новый параметр
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),  # Добавлен Dropout

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob / 2),  # Меньший Dropout на втором слое
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob),
            nn.Dropout2d(p=dropout_prob / 2)  # Дополнительный Dropout между блоками
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class _Up(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bilinear: bool = True,
            dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_prob)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob)

        self.dropout = nn.Dropout2d(p=dropout_prob / 2)  # Dropout после апсемплинга

    def forward(self, x: Tensor, x_skip: Tensor) -> Tensor:
        x = self.up(x)

        # Выравнивание размеров
        diff_h = x_skip.size()[2] - x.size()[2]
        diff_w = x_skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        return self.dropout(x)  # Применяем Dropout


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Dropout2d(p=0.1)  # Небольшой Dropout на выходе
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            bilinear: bool = True,
            init_features: int = 64,
            dropout_prob: float = 0.2,  # Основной параметр Dropout
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        # Encoder
        self.in_conv = DoubleConv(in_channels, init_features, dropout_prob=dropout_prob)
        self.down1 = _Down(init_features, init_features * 2, dropout_prob)
        self.down2 = _Down(init_features * 2, init_features * 4, dropout_prob)
        self.down3 = _Down(init_features * 4, init_features * 8, dropout_prob)

        # Bottleneck с увеличенным Dropout
        factor = 2 if bilinear else 1
        self.down4 = _Down(init_features * 8, init_features * 16 // factor, dropout_prob * 1.5)

        # Decoder
        self.up1 = _Up(init_features * 16, init_features * 8 // factor, bilinear, dropout_prob)
        self.up2 = _Up(init_features * 8, init_features * 4 // factor, bilinear, dropout_prob)
        self.up3 = _Up(init_features * 4, init_features * 2 // factor, bilinear, dropout_prob)
        self.up4 = _Up(init_features * 2, init_features, bilinear, dropout_prob)

        self.out_conv = OutConv(init_features, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)