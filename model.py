# ==== Model ====
# CLIP
# -> TCN
# -> LLM
# -> MLP
# ===============

import torch
import torch.nn as nn
import clip
from transformers import GPT2Model, GPT2Config


# CLIP Backbone
class CLIPBackbone(nn.Module):
    def __init__(self, model_name='ViT-B/32', device='cpu'):
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


# TemporalConvNet (Conv2d) that outputs flattened vector (B, T * (J * 3))
class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_joints=17, T=8):
        super().__init__()
        self.T = T
        self.num_joints = num_joints
        self.output_dim = num_joints * 3
        self.tcn = nn.Sequential(
            nn.Conv2d(input_size, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(2, 0), dilation=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, self.output_dim, kernel_size=(1, 1))
        )

    def forward(self, x):  # x: (B, T, D)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, D, T, 1)
        out = self.tcn(x)                     # (B, J*3, T, 1)
        out = out.squeeze(-1).permute(0, 2, 1)  # (B, T, J*3)
        return out.reshape(x.size(0), -1)       # (B, T * (J*3))


# GPT-2 wrapper to model flattened temporal joint sequences
class GPT2PoseWrapper(nn.Module):
    def __init__(self, input_dim, gpt2_dim=768, output_dim=768, T=8):
        super().__init__()
        self.T = T
        self.output_dim = output_dim

        self.project_in = nn.Linear(input_dim, gpt2_dim)
        self.project_out = nn.Linear(gpt2_dim, output_dim)

        config = GPT2Config(
            n_embd=gpt2_dim,
            n_layer=12,
            n_head=12,
            n_positions=1024,
            n_ctx=1024,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.gpt2 = GPT2Model(config)

        for param in self.gpt2.parameters():
            param.requires_grad = False  # freeze GPT-2

    def forward(self, x):  # x: (B, T * (J*3))
        x = self.project_in(x)           # (B, gpt2_dim)
        x = x.unsqueeze(1)               # (B, 1, gpt2_dim)
        x = self.gpt2(inputs_embeds=x).last_hidden_state  # (B, 1, gpt2_dim)
        return x.view(x.size(0), -1)


# Final Generator with reshaping logic included
class Generator(nn.Module):
    def __init__(self, input_dim, num_joints, T):
        super().__init__()
        self.linear = nn.Linear(input_dim, T*num_joints*3)
        self.T = T
        self.num_joints = num_joints

    def forward(self, x):  # x: (B, input_dim)
        B = x.size(0)
        x = self.linear(x)
        return x.view(B, self.T, self.num_joints, -1)


# Combined Pose Estimator Model
class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17, T=30, device='cuda'):
        super().__init__()
        self.num_joints = num_joints
        self.T = T

        self.backbone = CLIPBackbone(device=device)
        self.temporal = TemporalConvNet(input_size=self.backbone.output_dim, num_joints=num_joints, T=T)
        flat_output_dim = T * num_joints * 3
        self.llm_block = GPT2PoseWrapper(input_dim=flat_output_dim, output_dim=768, T=T)
        self.generator = Generator(input_dim=768, num_joints=num_joints, T=T)

    def forward(self, x):  # x: (B, T, C, H, W)
        features = self.backbone(x)             # (B, T, D)
        flat_pose = self.temporal(features)     # (B, T * (J*3))
        refined = self.llm_block(flat_pose)     # (B, T, J, 3)
        output = self.generator(refined)        # (B, T, J, 3)
        return output
