from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

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