from torchvision import datasets , transforms 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from LeNet5 import LeNet5 ,transform_of_LeNet , transform_of_LeNet_Normalize
from utils import *

import os
os.makedirs("Record",exist_ok=True)

 

# Dataset
train_dataset = datasets.MNIST( 'data/' , train=True  , transform=transform_of_LeNet ,download=True )
test_dataset  = datasets.MNIST( 'data/' , train=False , transform=transform_of_LeNet ,download=True )

train_loader = torch.utils.data.DataLoader( train_dataset , batch_size=32 , shuffle=True )
test_loader  = torch.utils.data.DataLoader( test_dataset  , batch_size=32 , shuffle=True )



# Model
model = LeNet5().to("cuda")

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Training :
num_epochs = 20

train_losses = []
train_accs   = []
test_losses  = []
test_accs    = []


for epoch in range(num_epochs):
    model.train() 
    correct = 0
    total = 0
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in pbar :
        images, labels = images.to("cuda"), labels.to("cuda")

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate  accuracy & loss
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update tqdm show
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")  

    
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_running_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
        test_epoch_loss = test_running_loss / len(test_loader.dataset)
        test_epoch_acc = 100 * test_correct / test_total
        test_losses.append(test_epoch_loss)
        test_accs.append(test_epoch_acc)
        print(f"Test  [{epoch+1}/{num_epochs}] - Loss: {test_epoch_loss:.4f} - Acc: {test_epoch_acc:.2f}%")
    model.train()

plot_training_curve(train_losses, train_accs, test_losses, test_accs)
torch.save(model.state_dict(), "model/LeNet5_weights.pth")



# Evaluation

model.eval()
correct = 0
total = 0
running_loss = 0.0

all_labels = []
all_preds  = []

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Testing")
    for images, labels in pbar:
        images, labels = images.to("cuda"), labels.to("cuda")

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate accuracy & loss
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update tqdm
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100 * correct / total:.2f}%"
        })

        all_labels.append(labels.cpu())
        all_preds.append(predicted.cpu())

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100 * correct / total:.2f}%"
        })

epoch_loss = running_loss / len(test_loader.dataset)
epoch_acc = 100 * correct / total
print(f" Test - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
plot_confusion_matrix(all_labels, all_preds, num_classes=10)





# Visualize Test Result

model_input, true_labels = next(iter(test_loader))   
model_input = model_input.to("cuda")

model.eval()
with torch.no_grad():
    predictions = model(model_input).argmax(dim=1)
visualize_predictions(model_input.cpu(), true_labels, predictions.cpu(),show=False,save=True,save_path="Record/Result_of_LeNet_Training.png")