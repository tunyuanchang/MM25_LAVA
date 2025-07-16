# ==== Model ====
# CLIP
# -> TCN
# -> LLM
# -> MLP
# ===============

import torch
import torch.nn as nn
from model.utils import VisionEmbedder, FoundationModel, Generator

# Temporal Embedder (TCN), Shape [B, T, D] -> [B, T * D]
class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x: [B, D, T]
        x = self.conv(x)               # [B, H, T]
        # x = x.permute(0, 2, 1)         # [B, T, H]
        x = self.norm(x)               # LayerNorm over feature dim
        x = self.relu(x)
        # x = x.permute(0, 2, 1)         # Back to [B, H, T]
        return x

class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, T=8):
        super().__init__()
        self.T = T
        self.output_dim = input_dim

        self.tcn = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, kernel_size=3, dilation=1),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.Conv1d(hidden_dim, self.output_dim, kernel_size=1)  # final projection
        )

    def forward(self, x):  # x: [B, T, D]
        B = x.size(0)
        x = x.permute(0, 2, 1)                # [B, D, T]
        out = self.tcn(x)                     # [B, output_dim, T]
        out = out.permute(0, 2, 1)            # [B, T, output_dim]
        return out.reshape(B, -1)

# Combined Pose Estimator Model
class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17, T=30, device='cuda'):
        super().__init__()
        self.num_joints = num_joints
        self.T = T

        self.visionembed = VisionEmbedder(device=device)
        self.D = self.visionembed.output_dim

        self.temporal = TCN(input_dim=self.D, T=self.T)
        self.llm = FoundationModel(input_dim=self.T*self.D, T=self.T)
        self.generator = Generator(input_dim=768, num_joints=num_joints, T=self.T)

    def forward(self, x):  # x: (B, T, C, H, W)
        x = self.visionembed(x)       # (B, T, D)
        x = self.temporal(x)          # (B, T * D)
        x = self.llm(x)               # (B, T, F)
        output = self.generator(x)    # (B, T, J, 3)
        return output
