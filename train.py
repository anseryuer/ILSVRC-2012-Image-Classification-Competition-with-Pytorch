# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.transforms import get_naive_transforms, get_val_transforms, target_transform
from utils.data_utils import get_train_dataset, get_val_dataset, get_test_dataset
from models.models import Small_Simple_Net, Res_Norm_Dropout_Net # Change this line to switch between models, or add more models to the import statement
from utils.training import train_one_epoch, validate, write_log
from utils.evaluation import evaluate, save_topk_indices


# %%

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE = 0.001
N_EPOCHS = 2
TEST_PERIOD = 1

# Set to True to train from a checkpoint, False to train from scratch.
TRAIN_FROM_CHECKPOINT = False
# If training from a checkpoint, set the paths to the checkpoint files here.
MODEL_CHECKPOINT_PATH = "models/your_model_checkpoint.pt"
OPTIMIZER_CHECKPOINT_PATH = "models/your_optimizer_checkpoint.pt"

# Set the path to the output log file here.
MODEL_CHECKPOINT_SAVE_DIR = "models/"
OPTIMIZER_CHECKPOINT_SAVE_DIR = "models/"
OUTPUT_LOG_DIR = "models/"

# Set root directory
root = r"/mnt/c/local_workplaces/ILSVRC 2012" # Change this line to the root directory of the project
train_dir = root + r"/dataset/ILSVRC2012_img_train"
val_dir = root + r"/dataset/ILSVRC2012_img_val"
test_dir = root + r"/dataset/test" # Change this line to the path of the test dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
device.type

# Load labels
with open(root + r"/dataset/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt") as f:
    labels = [line.strip() for line in f.readlines()]

meta_df = pd.read_csv(f"{root}/meta.csv")


# %%
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

# %%

# Load datasets
train_dataset = get_train_dataset(train_dir,get_naive_transforms(), lambda x: target_transform(x, meta_df))
val_dataset = get_val_dataset(val_dir, get_val_transforms(), labels)

# Load dataloaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"Data loaded, with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")


# %%
#plot(train_dataset[1][0:1])

# %%
#labels

# %%

# Load model
net = Small_Simple_Net().to(device) # Change this line to switch between models, or add more models to the import statement
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

print(summary(net, (3, 256, 256), TRAIN_BATCH_SIZE, device=device.type))
print(f"Model loaded, with {sum(p.numel() for p in net.parameters())} parameters")


# %%

# Train model
if TRAIN_FROM_CHECKPOINT:
    net.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_PATH))
    print("Model loaded from checkpoint")


train_log = {
    "train": {
        "epochs": [],
        "loss": [], # the loss is the average of the cross-entropy loss
        "acc": [], # the accuracy is the percentage of correct predictions
    },
    "validation": {
        "epochs": [], 
        "loss": [], # the loss is the average of the cross-entropy loss
        "acc": [] # the accuracy is the percentage of correct predictions
    }
}

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train_one_epoch(train_loader, device, net, optimizer, criterion, epoch)
        
    train_log["train"]["epochs"].append(epoch)
    train_log["train"]["loss"].append(train_loss)
    train_log["train"]["acc"].append(train_acc)
    
    torch.cuda.empty_cache()

    if epoch % TEST_PERIOD == TEST_PERIOD - 1:
        torch.save(net.state_dict(), MODEL_CHECKPOINT_PATH + f"model_{epoch}_{net._get_name()}.pt")
        torch.save(optimizer.state_dict(), OPTIMIZER_CHECKPOINT_PATH + f"optimizer_{epoch}_{net._get_name()}.pt")

        val_loss, val_acc = validate(val_loader, device, net, criterion)
        train_log["validation"]["epochs"].append(epoch)
        train_log["validation"]["loss"].append(val_loss)
        train_log["validation"]["acc"].append(val_acc)
        torch.cuda.empty_cache()

    write_log(train_log, OUTPUT_LOG_DIR + f"log_{net._get_name()}.json")

print("Training complete")

# Make predictions on test set
test_dataset = get_test_dataset(test_dir, get_val_transforms())
test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"Test dataset loaded, with {len(test_dataset)} samples")

# Load best model
best_epoch = train_log["validation"]["epochs"][train_log["validation"]["acc"].index(max(train_log["validation"]["acc"]))]
print(f"Loading model from epoch {best_epoch} with the best validation accuracy")
net.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH + f"model_{best_epoch}_{net._get_name()}.pt"))
topk_indices_list = evaluate(net, test_loader, device)
save_topk_indices(topk_indices_list, f"top5_indices_{net._get_name()}.txt")



