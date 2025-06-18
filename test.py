import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.tcn import PoseEstimator
from dataset import PoseDataset

ROOT_DIR = f"/media/tunyuan/Backup/Human36M/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 3
WINDOW_SIZE = 30

def mpjpe(predicted, target):
    # predicted, target: (B, T, J, 3)
    return torch.mean(torch.norm(predicted - target, dim=-1))


model = PoseEstimator(num_joints=16, device=device).to(device)

train_dataset = PoseDataset(ROOT_DIR, 'image_train', 'joint_train', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
valid_dataset = PoseDataset(ROOT_DIR, 'image_test', 'joint_test', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
print(len(train_dataset), len(valid_dataset))

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=8, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, pin_memory=True)
print(len(train_loader), len(valid_loader))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

torch.cuda.empty_cache()

print('Start Training')
for epoch in range(EPOCH):
    pbar = tqdm(train_loader)
    pbar.set_description(f"Training epoch [{epoch+1}]")

    model.train()
    total_loss = 0

    for (frames, keypoints_3d) in pbar:
        optimizer.zero_grad()

        frames = frames.to(device)           # (B, 30, C, H, W)
        keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)

        preds = model(frames)  # (B, 30, J, 3)
        loss = mpjpe(preds, keypoints_3d[:, :, 1:, :])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss:.4f}")

    print(f"Epoch {epoch + 1} | Training Loss: {total_loss / len(train_loader):.4f}")

    pbar = tqdm(valid_loader)
    pbar.set_description(f"Validation epoch [{epoch+1}]")

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for (frames, keypoints_3d) in pbar:
            frames = frames.to(device)           # (B, 30, C, H, W)
            keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)

            preds = model(frames)  # (B, 30, J, 3)
            loss = mpjpe(preds, keypoints_3d[:, :, 1:, :])

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss:.4f}")

    print(f"Epoch {epoch + 1} | Validation Loss: {total_loss / len(valid_loader):.4f}")

torch.save(model.state_dict(), "LLM-Skeleton3D.pth")