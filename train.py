import time
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
    parser.add_argument("--model", "--m", type=str, default="tcn", help="Model setting")
    parser.add_argument("--window_size", "--w", type=int, default=30, help="Window size of sliding window")
    parser.add_argument("--epoch", "--e", type=int, default=10, help="Number of epoch")

    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", "--b", type=int, default=4, help="Batch size")
    
    parser.add_argument("--checkpoint", "--c", type=str, default=None, help="Existing checkpoint")
    parser.add_argument("--input", "--i", type=str, default=None, help="Input root dir")
    parser.add_argument("--output", "--o", type=str, default=None, help="Output dir name")
    
    args = parser.parse_args()
    MODEL_NAME = args.model.lower()
    EPOCH = args.epoch
    WINDOW_SIZE = args.window_size
    LR = args.learning_rate
    BATCH_SIZE = args.batch_size
    INDEX = args.gpu_index
    CHECKPOINT = args.checkpoint
        
    if args.input != None:
        ROOT_DIR = args.input
    if args.output != None:
        RESULT_DIR = args.output

    # device
    if torch.cuda.is_available() and INDEX < torch.cuda.device_count():
        torch.cuda.set_device(INDEX)
        device = torch.device(f"cuda:{INDEX}")

    else:
        device = torch.device("cpu")

    print(device)

    # model
    match MODEL_NAME:
        case "tcn":
            model = TCN(num_joints=NUM_JOINTS-1, T=WINDOW_SIZE, device=device).to(device)
        case "gru":
            model = GRU(num_joints=NUM_JOINTS-1, T=WINDOW_SIZE, device=device).to(device)
        case "lstm":
            model = LSTM(num_joints=NUM_JOINTS-1, T=WINDOW_SIZE, device=device).to(device)
        case "concat":
            model = Concat(num_joints=NUM_JOINTS-1, T=WINDOW_SIZE, device=device).to(device)
        case "single":
            WINDOW_SIZE = 1
            model = Concat(num_joints=NUM_JOINTS-1, T=WINDOW_SIZE, device=device).to(device)
        case _:
            print("Error model name!")
            exit(0)

    print(MODEL_NAME)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    if CHECKPOINT != None and CHECKPOINT.endswith(".ckpt"):
        checkpoint = torch.load(f"{RESULT_DIR}{CHECKPOINT}", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # dataset
    train_dataset = PoseDataset(ROOT_DIR, 'image_train', 'joint_train', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
    test_dataset = PoseDataset(ROOT_DIR, 'image_test', 'joint_test', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    print(len(train_dataset), len(test_dataset))
    print(len(train_loader), len(test_loader))


    torch.cuda.empty_cache()

    # train & test

    train_loss = []
    test_loss = []

    train_time = []
    test_time = []

    for epoch in range(EPOCH):

        # training
        model.train()
        total_loss = 0
        start_time = time.time()  # Start timer

        count = 0

        for (frames, keypoints_3d) in train_loader:
            optimizer.zero_grad()

            frames = frames.to(device)           # (B, 30, C, H, W)
            keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)

            preds = model(frames)  # (B, 30, J, 3)
            loss = mpjpe(preds, keypoints_3d[:, :, 1:, :])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        duration = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"Training, Epoch {start_epoch + epoch + 1}, Loss: {avg_loss:.4f}, Time: {duration:.2f}")
        train_loss.append(avg_loss)
        train_time.append(duration)

        # testing
        model.eval()
        total_loss = 0
        start_time = time.time()  # Start timer

        with torch.no_grad():
            for (frames, keypoints_3d) in test_loader:
                frames = frames.to(device)           # (B, 30, C, H, W)
                keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)

                preds = model(frames)  # (B, 30, J, 3)
                loss = mpjpe(preds, keypoints_3d[:, :, 1:, :])

                total_loss += loss.item()
        
        duration = time.time() - start_time
        avg_loss = total_loss / len(test_loader)
        print(f"Testing, Epoch {start_epoch + epoch + 1}, Loss: {avg_loss:.4f}, Time: {duration:.2f}")
        test_loss.append(avg_loss)
        test_time.append(duration)

    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': start_epoch + EPOCH
    }, f'{RESULT_DIR}checkpoint_{MODEL_NAME}_{start_epoch + EPOCH}.ckpt')

    with open(f"{RESULT_DIR}train_time_{MODEL_NAME}_{start_epoch + EPOCH}.txt", 'w') as f:
        for epoch in range(EPOCH):
            print(f"{start_epoch + epoch + 1},{train_time[epoch]}", file=f)

    with open(f"{RESULT_DIR}test_time_{MODEL_NAME}_{start_epoch + EPOCH}.txt", 'w') as f:
        for epoch in range(EPOCH):
            print(f"{start_epoch + epoch + 1},{test_time[epoch]}", file=f)

    with open(f"{RESULT_DIR}train_loss_{MODEL_NAME}_{start_epoch + EPOCH}.txt", 'w') as f:
        for epoch in range(EPOCH):
            print(f"{start_epoch + epoch + 1},{train_loss[epoch]}", file=f)

    with open(f"{RESULT_DIR}test_loss_{MODEL_NAME}_{start_epoch + EPOCH}.txt", 'w') as f:
        for epoch in range(EPOCH):
            print(f"{start_epoch + epoch + 1},{test_loss[epoch]}", file=f)