import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv8 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1000)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.pool(F.relu(self.conv8(x)))
        x = torch.reshape(x, (x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class Small_Simple_Net(nn.Module):
    def __init__(self):
        super(Small_Simple_Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1000)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.reshape(x, (x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

class Res_Norm_Dropout_Net(nn.Module):
    def __init__(self):
        super(Res_Norm_Dropout_Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, padding="same")
        self.conv2 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv3 = nn.Conv2d(128, 256, 3, padding="same")
        self.conv4 = nn.Conv2d(256, 512, 3, padding="same")
        self.conv5 = nn.Conv2d(512, 512, 3, padding="same")
        self.conv6 = nn.Conv2d(512, 512, 3, padding="same")
        self.res_conv1 = nn.Conv2d(3, 128, 1, padding="same")
        self.res_conv2 = nn.Conv2d(128, 512, 1, padding="same")
        self.res_conv3 = nn.Conv2d(512, 512, 1, padding="same")
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.pool2 = nn.AvgPool2d(4, 4, padding=0)
        self.pool3 = nn.MaxPool2d(4, 4, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.res_batch_norm1 = nn.BatchNorm2d(128)
        self.res_batch_norm2 = nn.BatchNorm2d(512)
        self.res_batch_norm3 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1000)
        self.dropout = nn.Dropout(0.15)
    def forward(self,x):
        res1 = self.res_batch_norm1(self.pool2(F.relu(self.res_conv1(x)))) # 128 * 64 * 64
        x = self.pool1(F.relu(self.conv1(x))) # 64 * 128 * 128
        x = self.pool1(F.relu(self.conv2(x))) # 128 * 64 * 64
        x = self.batch_norm1(x) # 128 * 64 * 64
        x = x + res1 # 128 * 64 * 64
        res2 = self.res_batch_norm2(self.pool2(F.relu(self.res_conv2(x)))) # 512 * 16 * 16
        x = self.pool1(F.relu(self.conv3(x))) # 256 * 32 * 32
        x = self.pool1(F.relu(self.conv4(x))) # 512 * 16 * 16
        x = self.batch_norm2(x)
        x = x + res2 # 512 * 16 * 16
        res3 = self.res_batch_norm3(self.pool2(F.relu(self.res_conv3(x)))) # 512 * 4 * 4
        x = self.pool1(F.relu(self.conv5(x))) # 512 * 8 * 8
        x = self.pool1(F.relu(self.conv6(x))) # 512 * 4 * 4
        x = self.batch_norm3(x) # 512 * 4 * 4
        x = x + res3
        x = self.pool3(x) # 512 * 1 * 1
        x = torch.reshape(x, (x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x