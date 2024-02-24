import torch

def evaluate_acc(pred:torch.Tensor, target:torch.Tensor):
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)

def train_one_epoch(train_loader, device, net, optimizer, criterion, epoch, log_interval = 20):
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
        running_acc += evaluate_acc(outputs, targets)
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / (batch_idx + 1):.6f}\tAcc: {running_acc / (batch_idx + 1):.6f}")
    return running_loss / len(train_loader), running_acc / len(train_loader)