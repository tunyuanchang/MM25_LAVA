import torch
import torch.nn as nn
import clip
from transformers import GPT2Model

# CLIP (Frozen), Shape [B, T, C, H, W] -> [B, T, D]
class VisionEmbedder(nn.Module):
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        super().__init__()
        self.clip_model, preprocess = clip.load(model_name, device=device)
        self.clip_model = self.clip_model.float()
        self.visual = self.clip_model.encode_image
        self.output_dim = 512

    def forward(self, x):  # x: (B, T, C, H, W)
        self.clip_model.eval()
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)    # (B*T, C, H, W)
        x = self.visual(x)            # (B*T, D)
        x = x.view(B, T, -1)          # (B, T, D)
        return x

# GPT-2, Shape [B, T, D] -> [B, T, F]
class FoundationModel(nn.Module):
    def __init__(self, input_dim, F=768, T=30):
        super().__init__()
        self.T = T
        self.F = F
        self.project = nn.Linear(input_dim, F)
        self.gpt2 = GPT2Model.from_pretrained("openai-community/gpt2")

        for param in self.gpt2.parameters():
            param.requires_grad = False  # freeze GPT-2

    def forward(self, x):  # x: (B, T * D)
        B = x.size(0)

        x = x.reshape(B * self.T, -1)           # (B * T, D)
        x = self.project(x)                     # (B * T, F)
        x = x.reshape(B, self.T, -1)            # (B, T, F)
        
        x = self.gpt2(inputs_embeds=x).last_hidden_state  # (B, T, F)
        
        return x


# Generator (Neural Network), Shape [B, T, F] -> [B, T, J, 3]
class Generator(nn.Module):
    def __init__(self, input_dim, num_joints, T):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_joints*3)
        self.T = T
        self.num_joints = num_joints

    def forward(self, x):  # x: (B, T, F)
        B = x.size(0)

        x = x.reshape(B * self.T, -1)              # (B * T, F)
        x = self.linear(x)                      # (B * T, J * 3)
        
        return x.reshape(B, -1, self.num_joints, 3)    # (B, T, J, 3)