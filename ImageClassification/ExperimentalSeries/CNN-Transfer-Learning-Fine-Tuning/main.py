import torch
from model import ModelFactory
from dataset import get_dataloader
from trainer import Trainer
from utils import evaluate, save_txt
from visualize import plot_curve
from tqdm import tqdm   

device = "cuda" if torch.cuda.is_available() else "cpu"

# vgg16, resnet18, efficientnet, convnext
# (freeze / partial / full)
# False / True
MODEL_NAME = "efficientnet"      
STRATEGY = "full"          
AUGMENT = False

# Human Interface


train_path = "cats_and_dogs_small/train"
val_path = "cats_and_dogs_small/validation"
test_path = "cats_and_dogs_small/test"

train_loader = get_dataloader(train_path, augment=AUGMENT)
val_loader = get_dataloader(val_path)
test_loader = get_dataloader(test_path)

name = f"{MODEL_NAME}_{STRATEGY}"
print(f"Start : {name}\n")

model = ModelFactory(MODEL_NAME, strategy=STRATEGY).build()
trainer = Trainer(model, train_loader, val_loader, device, name)
acc, total_time = trainer.train(epochs=10)
print(f"Validation accuracy: {acc:.4f}")

# after training
best_model_path = f"results/{name}.pth"
model.load_state_dict(torch.load(best_model_path,weights_only=True))
test_acc = evaluate(model, test_loader, device)
print(f"Test accuracy: {test_acc:.4f}")
 
save_txt(f"results/{name}_record.txt", f"Test Acc : {test_acc};\nValid Acc : {acc};\nTime Cost : {total_time};\n")
plot_curve(f"Record/{name}.csv")

print(f"\nComplete")
print(f"Total training time : {total_time:.2f} s")