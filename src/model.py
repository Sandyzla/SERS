import torch
import torch.nn as nn
import numpy as np


class LearnedSpectralMask(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.mask_weights = nn.Parameter(torch.ones(1, 1, input_len))

    def forward(self, x):
        return x * self.mask_weights


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        # self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=25, padding=12)
        self.relu = nn.ReLU()
        self.fusion = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        # b3 = self.relu(self.branch3(x))
        return self.fusion(torch.cat([b1, b2], dim=1))


class CoreModel(nn.Module):
    def __init__(self, input_len=1201, num_classes=3):
        super().__init__()
        self.spectral_mask = LearnedSpectralMask(input_len)
        self.cnn_frontend = nn.Sequential(
            MultiScaleBlock(1, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, 64))  # 预估长度，稍微大一点防止溢出
        self._to_linear = None
        self._check_dimension(input_len)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def _check_dimension(self, input_len):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_len)
            x = self.cnn_frontend(self.spectral_mask(x))
            x = x.permute(0, 2, 1)
            self._to_linear = int(np.prod(x.size()))

    def forward(self, x):
        x = self.cnn_frontend(self.spectral_mask(x))
        x = x.permute(0, 2, 1)  # [Batch, Seq, Feature]
        # 动态截取位置编码以适应可能的尺寸变化
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return self.classifier(x)