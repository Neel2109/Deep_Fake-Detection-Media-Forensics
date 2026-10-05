from __future__ import annotations

from typing import Any


def create_model(torch: Any) -> Any:
    nn = torch.nn

    class SeparableConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            )
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )

        def forward(self, value: Any) -> Any:
            return self.pointwise(self.depthwise(value))

    class XceptionBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            repetitions: int,
            stride: int = 1,
            start_with_relu: bool = True,
            use_pooling: bool = True,
        ) -> None:
            super().__init__()
            layers = []
            for index in range(repetitions):
                input_channels = in_channels if index == 0 else out_channels
                layers.extend([
                    nn.ReLU() if start_with_relu or index > 0 else nn.Identity(),
                    SeparableConv2d(input_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                ])
            if use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))
            self.layers = nn.ModuleList(layers)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, value: Any) -> Any:
            residual = self.shortcut(value)
            for layer in self.layers:
                value = layer(value)
            return value + residual

    class Xception(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()

            self.block1 = XceptionBlock(64, 128, repetitions=2, stride=2, start_with_relu=False)
            self.block2 = XceptionBlock(128, 256, repetitions=2, stride=2)
            self.block3 = XceptionBlock(256, 728, repetitions=2, stride=2)
            self.middle_flow = nn.Sequential(
                *(XceptionBlock(728, 728, repetitions=3, use_pooling=False) for _ in range(8))
            )
            self.block4 = XceptionBlock(728, 1024, repetitions=2, stride=2)
            self.sepconv1 = SeparableConv2d(1024, 1536)
            self.bn3 = nn.BatchNorm2d(1536)
            self.sepconv2 = SeparableConv2d(1536, 2048)
            self.bn4 = nn.BatchNorm2d(2048)
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(2048, 2)

        def forward(self, value: Any) -> Any:
            value = self.relu(self.bn1(self.conv1(value)))
            value = self.relu(self.bn2(self.conv2(value)))
            value = self.block1(value)
            value = self.block2(value)
            value = self.block3(value)
            value = self.middle_flow(value)
            value = self.block4(value)
            value = self.relu(self.bn3(self.sepconv1(value)))
            value = self.relu(self.bn4(self.sepconv2(value)))
            value = self.global_avg_pool(value)
            value = value.reshape(value.shape[0], -1)
            return self.fc(value)

    return Xception()
