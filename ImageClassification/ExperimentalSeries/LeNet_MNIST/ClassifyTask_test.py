from torchvision import datasets , transforms 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from LeNet5 import LeNet5 ,transform_of_LeNet , transform_of_LeNet_Normalize
from utils import *



# Dataset
test_dataset  = datasets.MNIST( 'data/' , train=False , transform=transform_of_LeNet ,download=True )
test_loader  = torch.utils.data.DataLoader( test_dataset  , batch_size=32 , shuffle=True )
criterion = nn.CrossEntropyLoss()


model = LeNet5().to("cuda")
model.load_state_dict(torch.load("model/LeNet5_weights.pth",weights_only=True))
model.eval()  


# Test

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
plot_confusion_matrix(all_labels, all_preds, num_classes=10,save_path="_ConfusionMatrix.png")





# Visualize Test Result

model_input, true_labels = next(iter(test_loader))   
model_input = model_input.to("cuda")

model.eval()
with torch.no_grad():
    predictions = model(model_input).argmax(dim=1)
visualize_predictions(model_input.cpu(), true_labels, predictions.cpu(),show=False,save=True,save_path="_Result_of_LeNet_Training.png")