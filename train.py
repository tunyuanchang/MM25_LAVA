import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.concat import PoseEstimator as Concat
from model.tcn import PoseEstimator as TCN
from model.gru import PoseEstimator as GRU
from model.lstm import PoseEstimator as LSTM
from dataset import PoseDataset

# config
ROOT_DIR = "./Human36M/"
RESULT_DIR = "./result/"
NUM_JOINTS = 17

# loss function
def mpjpe(predicted, target):
    # predicted, target: (B, T, J, 3)
    return torch.mean(torch.norm(predicted - target, dim=-1))

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_index", "--g", type=int, default=0, help="GPU index")
    parser.add_argument("--model", "--m", type=str, default="TCN", help="Model setting")
    parser.add_argument("--window_size", "--w", type=int, default=30, help="Window size of sliding window")
    parser.add_argument("--batch_size", "--b", type=int, default=4, help="Batch size")
    parser.add_argument("--epoch", "--e", type=int, default=10, help="Number of epoch")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--input", "--i", type=str, default=None, help="Input root dir")
    parser.add_argument("--output", "--o", type=str, default=None, help="Output dir name")
    
    args = parser.parse_args()
    MODEL_NAME = args.model
    EPOCH = args.epoch
    WINDOW_SIZE = args.window_size
    LR = args.learning_rate
    BATCH_SIZE = args.batch_size
    INDEX = args.gpu_index

    if args.input != None:
        ROOT_DIR = args.input
    if args.output != None:
        OUTPUT_DIR = f"{args.output}{MODEL_NAME}"
    else:
        OUTPUT_DIR = f"{RESULT_DIR}{MODEL_NAME}"

    # device
    if torch.cuda.is_available() and INDEX < torch.cuda.device_count():
        torch.cuda.set_device(INDEX)
        device = torch.device(f"cuda:{INDEX}")

    else:
        device = torch.device("cpu")

    print(device)

    # dataset
    train_dataset = PoseDataset(ROOT_DIR, 'image_train', 'joint_train', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
    valid_dataset = PoseDataset(ROOT_DIR, 'image_test', 'joint_test', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    print(len(train_dataset), len(valid_dataset))
    print(len(train_loader), len(valid_loader))

    # model
    if MODEL_NAME in ["TCN", "tcn"]:
        model = TCN(num_joints=NUM_JOINTS-1, device=device).to(device)
    
    elif MODEL_NAME in ["GRU", "gru"]:
        model = GRU(num_joints=NUM_JOINTS-1, device=device).to(device)

    elif MODEL_NAME in ["LSTM", "lstm"]:
        model = LSTM(num_joints=NUM_JOINTS-1, device=device).to(device)

    elif MODEL_NAME in ["Concat", "concat"]:
        model = Concat(num_joints=NUM_JOINTS-1, device=device).to(device)

    else:
        print("Error model name!")
        exit(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    torch.cuda.empty_cache()

    # train & valid

    train_loss = []
    valid_loss = []

    train_time = []
    valid_time = []

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
        train_loss.append(total_loss / len(train_loader))

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
        valid_loss.append(total_loss / len(valid_loader))

    torch.save(model.state_dict(), f"{OUTPUT_DIR}_model_{EPOCH}.pth")

    with open(f"{OUTPUT_DIR}_loss_{EPOCH}.txt", 'w') as f:
        print(','.join(map(str, train_loss)), file=f)
        print(','.join(map(str, valid_loss)), file=f)