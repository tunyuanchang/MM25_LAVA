# ==== Model ====
# CLIP
# -> Concatenation
# -> LLM
# -> MLP
# ===============

import torch
import torch.nn as nn
from model.utils import VisionEmbedder, FoundationModel, Generator

# Concatenation, Shape [B, T, D] -> [B, T * D]
class Concatenation(nn.Module):
    def __init__(self, input_dim, T=30):
        super().__init__()
        self.T = T

    def forward(self, x):  # x: (B, T, D)
        B = x.size(0)
        return x.reshape(B, -1)

# Combined Pose Estimator Model
class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17, T=30, device='cuda'):
        super().__init__()
        self.num_joints = num_joints
        self.T = T

        self.visionembed = VisionEmbedder(device=device)
        self.D = self.visionembed.output_dim

        self.temporal = Concatenation(input_dim=self.D, T=self.T)
        self.llm = FoundationModel(input_dim=self.T*self.D, T=self.T)
        self.generator = Generator(input_dim=768, num_joints=num_joints, T=self.T)

    def forward(self, x):  # x: (B, T, C, H, W)
        x = self.visionembed(x)       # (B, T, D)
        x = self.temporal(x)          # (B, T * D)
        x = self.llm(x)               # (B, T, F)
        output = self.generator(x)    # (B, T, J, 3)
        return output
