import os
import time
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.concat import PoseEstimator as Concat
from model.tcn import PoseEstimator as TCN
from model.gru import PoseEstimator as GRU
from model.lstm import PoseEstimator as LSTM
from dataset_test import PoseDataset
from collections import defaultdict

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
    parser.add_argument("--batch_size", "--b", type=int, default=16, help="Batch size")
    
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
    test_dataset = PoseDataset(ROOT_DIR, 'image_test', 'joint_test', window_size=WINDOW_SIZE, stride=WINDOW_SIZE)

    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
    


    torch.cuda.empty_cache()

    # train & test
    model.eval()
    total_loss = 0
    action_errors = defaultdict(list)

    with torch.no_grad():
        for (ids, frames, keypoints_3d) in test_loader:
            act_ids = ids
            frames = frames.to(device)           # (B, 30, C, H, W)
            keypoints_3d = keypoints_3d.to(device)  # (B, 30, J, 3)
            preds = model(frames)  # (B, 30, J, 3)
            
            for i in range(len(act_ids)):
            
                action = int(act_ids[i])
                pred = preds[i]                       # (30, J, 3)
                gt = keypoints_3d[i, :, 1:, :]        # Exclude root joint (30, J-1, 3)

                error = mpjpe(pred[:, :, :], gt)     # compute MPJPE for one sequence
                action_errors[action].append(error.item())

    average_error_per_action = {action: sum(errors)/len(errors) for action, errors in action_errors.items()}

    try:
        basename = os.path.basename(CHECKPOINT) # f'checkpoint_{MODEL_NAME}_{EPOCH}_{WINDOW_SIZE}.ckpt'
        _, model_name, epoch, window_size = (os.path.splitext(basename)[0]).split('_')
    except:
        model_name = MODEL_NAME
        epoch = EPOCH
        window_size = WINDOW_SIZE

    with open(f"{RESULT_DIR}test_action_{model_name}_{epoch}_{window_size}.txt", 'w') as f:
        for action, avg_error in average_error_per_action.items():
            print(f"{action}: {avg_error:.2f}", file=f)