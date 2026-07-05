import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ====================== Params ======================
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DATA_ROOT = "Data"
CONFIG_PATH = os.path.join(DATA_ROOT, "stage_config.json")

os.makedirs("model_pretrain", exist_ok=True)
os.makedirs("Record", exist_ok=True)

SAVE_PATH = os.path.join("model_pretrain", "stage01_model.pth")
RECORD_PATH = os.path.join("Record", "stage01_record.csv")

# ====================== Transform ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====================== Load config ======================
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    stage_config = json.load(f)

stage01_classes = stage_config["stage01"]
print(f"Stage 01 classes: {stage01_classes}")


def main():
    # ====================== Train Dataset ======================
    train_path = os.path.join(DATA_ROOT, "train")
    full_train_dataset = datasets.ImageFolder(train_path, transform=transform)

    # 只保留 stage01 的類別
    allowed_indices = []
    for idx, (path, label) in enumerate(full_train_dataset.samples):
        cls_name = full_train_dataset.classes[label]
        if cls_name in stage01_classes:
            allowed_indices.append(idx)

    train_dataset = Subset(full_train_dataset, allowed_indices)
    train_dataset.classes = stage01_classes
    train_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(stage01_classes)}

    print(f"Stage 01 training samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,      
        pin_memory=True
    )

    # ====================== Test Loader ======================
    def get_test_dataset(class_names, test_root, transform):
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        samples = []
        for cls in class_names:
            cls_path = os.path.join(test_root, cls)
            if os.path.isdir(cls_path):
                for fname in sorted(os.listdir(cls_path)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        path = os.path.join(cls_path, fname)
                        samples.append((path, class_to_idx[cls]))
        class TestDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform=None):
                self.samples = samples
                self.transform = transform
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                path, target = self.samples[idx]
                image = Image.open(path).convert('RGB')    
                if self.transform is not None:
                    image = self.transform(image)
                return image, target
        return TestDataset(samples, transform)

    def get_test_loader(class_names, test_root, transform, batch_size=32):
        dataset = get_test_dataset(class_names, test_root, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return dataset, loader

    from PIL import Image    

    test_root = os.path.join(DATA_ROOT, "test")
    test_dataset, test_loader = get_test_loader(stage01_classes, test_root, transform, BATCH_SIZE)

    # ====================== Evaluate ======================
    def evaluate(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    # ====================== Model ======================
    print("=== Stage 1 Training ===")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(stage01_classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    records = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        acc = evaluate(model, test_loader)
        records.append({
            'epoch': epoch + 1,
            'overall': round(acc, 4),
            'old': round(acc, 4),
            'new': round(acc, 4)
        })
        print(f"Epoch {epoch+1:2d} → Overall: {acc:.4f}")

    train_time = time.time() - start_time
    print(f"\nStage 1 Training Time: {train_time:.2f} sec")

    pd.DataFrame(records).to_csv(RECORD_PATH, index=False)
    print(f"Stage 1 record saved → {RECORD_PATH}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(stage01_classes),
        'class_names': stage01_classes,
        'stage_acc': records[-1]['overall']
    }, SAVE_PATH)
    print(f"Stage 1 model saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()