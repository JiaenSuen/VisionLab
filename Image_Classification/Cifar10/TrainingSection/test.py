import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from hyperparams import *
from utils import check_accuracy,save_model,load_model
from models import *

model = build_resnet18(num_classes=num_classes)
model = load_model(model, "modelp/resnet18.pth")


test_dataset = datasets.CIFAR10(
    root="dataset/", train=False, transform=test_transform, download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
check_accuracy(test_loader, model)