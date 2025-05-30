# ==== Model ====
# CLIP
# -> Concatenation
# -> LLM
# -> MLP
# ===============

import torch
import torch.nn as nn
import clip
from transformers import GPT2Model

# CLIP Backbone (Frozen), Shape [B, T, C, H, W] -> [B, T, D]
class CLIP(nn.Module):
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        super().__init__()
        self.clip_model, preprocess = clip.load(model_name, device=device)
        self.clip_model = self.clip_model.float()
        self.visual = self.clip_model.encode_image
        self.output_dim = 512

    def forward(self, x):  # x: (B, T, C, H, W)
        self.clip_model.eval()
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.visual(x)  # (B*T, D)
        x = x.view(B, T, -1)  # (B, T, D)
        return x


# Concatenation, Shape [B, T, D] -> [B, T * D]
class Concat(nn.Module):
    def __init__(self, input_dim, T=8):
        super().__init__()
        self.T = T

    def forward(self, x):  # x: (B, T, D)
        return x.reshape(x.size(0), -1)    # [B, T * D]


# GPT-2 Foundation Model, Shape [B, T * D] -> [B, F]
class GPT2PoseWrapper(nn.Module):
    def __init__(self, input_dim, F=768, T=8):
        super().__init__()
        self.T = T

        self.project_in = nn.Linear(input_dim, F)

        self.gpt2 = GPT2Model.from_pretrained("openai-community/gpt2")

        for param in self.gpt2.parameters():
            param.requires_grad = False  # freeze GPT-2

    def forward(self, x):  # x: (B, T * D)
        x = self.project_in(x)           # (B, F)
        x = x.unsqueeze(1)               # (B, 1, F)
        x = self.gpt2(inputs_embeds=x).last_hidden_state  # (B, 1, F)
        return x.view(x.size(0), -1)


# Generator (Neural Network), Shape [B, F] -> [B, T, J, 3]
class Generator(nn.Module):
    def __init__(self, input_dim, num_joints, T):
        super().__init__()
        self.linear = nn.Linear(input_dim, T*num_joints*3)
        self.T = T
        self.num_joints = num_joints

    def forward(self, x):  # x: (B, F)
        B = x.size(0)
        x = self.linear(x)
        return x.view(B, self.T, self.num_joints, -1)


# Combined Pose Estimator Model
class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17, T=30, device='cuda'):
        super().__init__()
        self.num_joints = num_joints
        self.T = T

        self.backbone = CLIP(device=device)
        self.D = self.backbone.output_dim

        self.temporal = Concat(input_dim=self.D, T=self.T)
        self.llm_block = GPT2PoseWrapper(input_dim=self.T * self.D, T=self.T)
        self.generator = Generator(input_dim=768, num_joints=num_joints, T=T)

    def forward(self, x):  # x: (B, T, C, H, W)
        features = self.backbone(x)             # (B, T, D)
        flat_pose = self.temporal(features)     # (B, T * D)
        refined = self.llm_block(flat_pose)     # (B, T, F)
        output = self.generator(refined)        # (B, T, J, 3)
        return output
