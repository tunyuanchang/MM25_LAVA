import torch
from torch.utils.data import DataLoader
from model import PoseEstimator
from dataset import PoseDataset

IMAGE_FOLDER = f"/media/tunyuan/Backup/Human36M/images/"
JOINT_FOLDER = f"/media/tunyuan/Backup/Human36M/annotations/"

def mpjpe(predicted, target):
    # predicted, target: (B, T, J, 3)
    return torch.mean(torch.norm(predicted - target, dim=-1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseEstimator(num_joints=16, device=device).to(device)

train_dataset = PoseDataset(IMAGE_FOLDER, JOINT_FOLDER, S=[1,5,6,7,8])
valid_dataset = PoseDataset(IMAGE_FOLDER, JOINT_FOLDER, S=[9])
print(len(train_dataset), len(valid_dataset))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)
# dataset = PoseDataset(IMAGE_FOLDER, JOINT_FOLDER)
# loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
print(len(train_loader), len(valid_loader))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

torch.cuda.empty_cache()

print('Start Training')
for epoch in range(10):
    model.train()
    total_loss = 0
    for i, (frames, keypoints_3d) in enumerate(train_loader):
        frames = frames.to(device)           # (B, 30, C, H, W)
        keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)

        preds = model(frames)  # (B, 30, J, 3)
        loss = mpjpe(preds, keypoints_3d[:, :, 1:, :])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} | Training Loss: {total_loss / len(train_loader):.4f}")

    
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (frames, keypoints_3d) in enumerate(valid_loader):
            frames = frames.to(device)           # (B, 30, C, H, W)
            keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)

            preds = model(frames)  # (B, 30, J, 3)
            loss = mpjpe(preds, keypoints_3d[:, :, 1:, :])

            total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Validation Loss: {total_loss / len(valid_loader):.4f}")

torch.save(model.state_dict(), "LLM-Skeleton3D.pth")