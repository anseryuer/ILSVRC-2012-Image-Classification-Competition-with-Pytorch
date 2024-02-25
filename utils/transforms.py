from torchvision.transforms import v2
import torch
import pandas as pd
def get_fancy_transforms():
    fancy_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        #v2.Resize(size=(256, 256)),
        #v2.CenterCrop(size=(256, 256)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=15),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
    ])
    return fancy_transforms

def get_naive_transforms():
    naive_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        #v2.Resize(size=(256, 256)),
        #v2.RandomCrop(size=(512, 512), pad_if_needed=True),
        v2.Resize(size=(256, 256)),
        v2.RandomHorizontalFlip(p=0.5),
        #v2.RandomVerticalFlip(p=0.3),
        #v2.RandomRotation(degrees=15),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
    ])

    return naive_transforms

def get_val_transforms():
    val_transforms = v2.Compose([
        v2.Resize(size=(256)),
        v2.CenterCrop(size=(256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
    ])
    return val_transforms

def target_transform(target, meta_df : pd.DataFrame):
    return meta_df.sort_values("WNID").reset_index().at[int(target), "ILSVRC2012_ID"] - 1 # -1 because the labels are 1-indexed, should be 0-indexed for crossentropy
