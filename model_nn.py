# ==== Model ====
# Resnet50
# -> TCN
# ===============

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, out_dim=2048):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feat = self.feature_extractor(x)
        feat = self.pool(feat).view(B, T, -1)  # (B, T, D)
        return feat

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_joints=17):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, num_joints * 3, kernel_size=1)
        )

    def forward(self, x):  # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        out = self.tcn(x)       # (B, J*3, T)
        out = out.permute(0, 2, 1)  # (B, T, J*3)
        return out.view(out.size(0), out.size(1), -1, 3)  # (B, T, J, 3)

class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.temporal = TemporalConvNet(input_size=2048, num_joints=num_joints)

    def forward(self, x):  # (B, T, C, H, W)
        features = self.backbone(x)         # (B, T, D)
        keypoints_3d = self.temporal(features)  # (B, T, J, 3)
        return keypoints_3d
