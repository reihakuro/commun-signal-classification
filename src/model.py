import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    1. Squeeze:  GlobalAvgPool   (B,C,H,W) → (B,C)
    2. Excite:   FC(C→mid) → ReLU → FC(mid→C) → Sigmoid  → (B,C,1,1)
    3. Scale:    element-wise multiply input by gates  → (B,C,H,W)

    Minimum bottleneck = max(C//reduction, 8) to avoid information loss
    on small channel widths.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)  # squeeze
        s = self.sigmoid(self.fc2(self.relu(self.fc1(s)))).view(b, c, 1, 1)  # excite
        return x * s  # scale


class InvertedResidualSE(nn.Module):
    """
    MobileNetV2 Inverted Residual block with optional SE attention.

    Layers (when expand_ratio > 1):
      Expand:  Conv(inp→hidden, 1×1) + BN + ReLU6
      DWConv:  Conv(hidden→hidden, 3×3, groups=hidden) + BN + ReLU6
      Project: Conv(hidden→oup, 1×1) + BN   ← linear, no activation
      SE:      SEBlock(oup)
      Skip:    x + out  (only when stride==1 and inp==oup)
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        use_se: bool = True,
        se_reduction: int = 4,
    ):
        super().__init__()
        hidden = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(inp, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ]
        layers += [
            nn.Conv2d(hidden, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(oup, se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.conv(x))
        if self.use_res_connect:
            return x + out
        return out


class BasicCNN(nn.Module):
    """
    RF Spectrogram Classifier: MobileNetV2 backbone + SE attention.

    Input : (batch_size, 3, 224, 224)
    Output: (batch_size, num_classes)  <- raw logits, no softmax
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.Sequential(
            InvertedResidualSE(16, 16, stride=1, expand_ratio=1, use_se=False),
            InvertedResidualSE(16, 24, stride=2, expand_ratio=6, use_se=True),
            InvertedResidualSE(24, 24, stride=1, expand_ratio=6, use_se=True),
            InvertedResidualSE(24, 32, stride=2, expand_ratio=6, use_se=True),
            InvertedResidualSE(32, 32, stride=1, expand_ratio=6, use_se=True),
            InvertedResidualSE(32, 48, stride=2, expand_ratio=6, use_se=True),
            InvertedResidualSE(48, 64, stride=2, expand_ratio=4, use_se=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling -> (B,128,1,1)
        )

        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # (B,  3, 224, 224) -> (B, 16, 112, 112)
        x = self.blocks(x)  #                   -> (B, 64,   7,   7)
        x = self.head(x)  #                   -> (B,128,   1,   1)
        x = x.view(x.size(0), -1)  # flatten           -> (B, 128)
        x = self.dropout(x)
        x = self.classifier(x)  #                   -> (B, num_classes)
        return x  # raw logits – required by project spec
