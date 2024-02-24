from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import os

def get_train_dataset(root, transform, target_transform):
    return ImageFolder(root, transform=transform, target_transform=target_transform)

class val_Dataset(Dataset):
    def __init__(self, image_dir , labels, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_dir + "/ILSVRC2012_val_" + str(idx + 1).zfill(8) + ".JPEG"
        img = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.labels[idx]
        label = int(label) - 1 # 0-indexed
        if self.transform:
            img = self.transform(img)
        return img, label
    
def get_val_dataset(root, transform, labels):
    return val_Dataset(root, labels, transform=transform)

class test_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_path = self.image_dir + "/ILSVRC2012_test_" + str(idx + 1).zfill(8) + ".JPEG"
        img = read_image(img_path, mode=ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img

def get_test_dataset(root, transform):
    return test_Dataset(root, transform=transform)