import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.transforms import naive_transforms, val_transforms, target_transform
from utils.data_utils import get_train_dataset, get_val_dataset
from models.models import Simple_Net, Small_Simple_Net, Res_Norm_Dropout_Net # Change this line to switch between models, or add more models to the import statement

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE = 0.001
N_EPOCHS = 80
TEST_PERIOD = 5

# Set to True to train from a checkpoint, False to train from scratch.
TRAIN_FROM_CHECKPOINT = False
# If training from a checkpoint, set the paths to the checkpoint files here.
MODEL_CHECKPOINT_PATH = "output/checkpoint.pth"
OPTIMIZER_CHECKPOINT_PATH = "output/optimizer.pth"

# Set root directory
root = r"/mnt/c/local_workplaces/ILSVRC 2012" # Change this line to the root directory of the project
train_dir = root + r"/dataset/ILSVRC2012_img_train"
val_dir = root + r"/dataset/ILSVRC2012_img_val"
test_dir = root + r"/dataset/test" # Change this line to the path of the test dataset

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    device.type

    # Load labels
    with open(root + r"/dataset/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    meta_df = pd.read_csv(f"{root}/meta.csv")

    # Load datasets
    train_dataset = get_train_dataset(train_dir, naive_transforms(), target_transform)
    val_dataset = get_val_dataset(val_dir, val_transforms(), labels)

    # Load dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Data loaded, with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # Load model
    net = Small_Simple_Net().to(device) # Change this line to switch between models, or add more models to the import statement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print(summary(net, (3, 256, 256), TRAIN_BATCH_SIZE, device=device.type))
    print(f"Model loaded, with {sum(p.numel() for p in net.parameters())} parameters")

    # Train model
    if TRAIN_FROM_CHECKPOINT:
        net.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
        optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_PATH))
        print("Model loaded from checkpoint")

    step_loss_log = []
    step_acc_log = []
    train_log = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(N_EPOCHS):
        ...

if __name__ == "__main__":
    main()