import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.optim as optim
import os

import numpy as np

def evaluate(net, test_dataset_loader, device):
    topk_indices_list = np.array([])
    with torch.no_grad():
        for i, data in enumerate(test_dataset_loader):
            inputs = data.to(device)
            outputs = net(inputs)
            _, topk_indices = torch.topk(outputs, k=5, dim=1)
            topk_indices_list = np.concatenate([topk_indices_list,topk_indices.cpu().numpy().reshape(-1)])

    topk_indices_list = topk_indices_list.reshape(-1,5)
    topk_indices_list = topk_indices_list.astype(np.int32)
    topk_indices_list = topk_indices_list + 1 # 1-indexed
    return topk_indices_list

def evaluate_onnx(onnx_model, test_dataset_loader, device):
    import onnx
    import onnxruntime
    import numpy as np

    topk_indices_list = np.array([])
    sess = onnxruntime.InferenceSession(onnx_model)
    for i, data in enumerate(test_dataset_loader):
        inputs = data.to(device)
        inputs = inputs.cpu().numpy()
        outputs = sess.run(None, {sess.get_inputs()[0].name: inputs})
        topk_indices = np.argsort(outputs, axis=1)[:,-5:]
        topk_indices_list = np.concatenate([topk_indices_list,topk_indices])

    topk_indices_list = topk_indices_list.reshape(-1,5)
    topk_indices_list = topk_indices_list.astype(np.int32)
    topk_indices_list = topk_indices_list + 1 # 1-indexed
    return topk_indices_list

def save_topk_indices(topk_indices_list, filename):
    with open(filename, "w") as f:
        for result in topk_indices_list:
            f.write(" ".join(result.astype(str)) + "\n")
    print(f"Top-5 indices saved to {filename}")