# ==== Model ====
# CLIP
# -> GRU
# -> LLM
# -> MLP
# ===============

import torch
import torch.nn as nn
from model.utils import VisionEmbedder, FoundationModel, Generator

# Temporal Embedder (GRU), Shape [B, T, D] -> [B, T * D]
class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)           # out: [B, T, hidden_size]
        return out                      # return full sequence

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, T=8):
        super().__init__()
        self.T = T
        self.output_dim = input_dim
        self.gru = nn.Sequential(
            GRUBlock(input_dim, hidden_dim),
            nn.ReLU(),
            GRUBlock(hidden_dim, self.output_dim)
        )

    def forward(self, x):  # x: [B, T, D]
        out = self.gru(x)              # [B, T, output_dim]
        return out

# Combined Pose Estimator Model
class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17, T=30, device='cuda'):
        super().__init__()
        self.num_joints = num_joints
        self.T = T

        self.visionembed = VisionEmbedder(device=device)
        self.D = self.visionembed.output_dim

        self.temporal = GRU(input_dim=self.D, T=self.T)
        self.llm = FoundationModel(input_dim=self.D, T=self.T)
        self.generator = Generator(input_dim=768, num_joints=num_joints, T=self.T)

    def forward(self, x):  # x: (B, T, C, H, W)
        x = self.visionembed(x)       # (B, T, D)
        x = self.temporal(x)          # (B, T * D)
        x = self.llm(x)               # (B, T, F)
        output = self.generator(x)    # (B, T, J, 3)
        return output
