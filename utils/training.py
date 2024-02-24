import torch
import json
from torch.utils.data import Dataset, DataLoader

def evaluate_acc(pred:torch.Tensor, target:torch.Tensor):
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)

def evaluate_loss(pred:torch.Tensor, target:torch.Tensor):
    return torch.nn.functional.cross_entropy(pred, target).item()

def train_one_epoch(train_loader : DataLoader, device, net, optimizer, criterion, epoch):
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc += evaluate_acc(outputs, targets) * len(targets)
    print(f"Epoch {epoch}: Average loss: {running_loss / len(train_loader.dataset):.4f}, Accuracy: {running_acc / len(train_loader.dataset):.4f}")
    return running_loss / len(train_loader.dataset), running_acc / len(train_loader.dataset)

def validate(val_loader : DataLoader, device, net, criterion):
    val_loss = 0
    val_acc = 0
    net.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            val_loss += evaluate_loss(outputs, targets)
            val_acc += evaluate_acc(outputs, targets) * len(targets)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print(f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}\n")
    return val_loss, val_acc

def write_log(dict1:dict, path:str):
    with open(path, "w") as f:
        json.dump(dict1, f, indent=4)  # Add "indent=4" for better readability
