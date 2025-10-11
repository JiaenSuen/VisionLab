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


model = build_resnet18(num_classes=10)
model.to(device)
#print(model)



train_dataset = torchvision.datasets.CIFAR10(
    root="dataset/", train=True, transform=train_transform, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    losses = []

    for batch_idx , (data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss   = criterion(scores,targets)

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Cost at epoch {epoch+1} is {sum(losses)/len(losses):.5f}")

check_accuracy(train_loader, model)
save_model(model, "trained/resnet18.pth")
