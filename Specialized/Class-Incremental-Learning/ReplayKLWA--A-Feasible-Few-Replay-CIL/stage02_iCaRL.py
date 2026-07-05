import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_STAGE = "02"         
BATCH_SIZE = 32
EPOCHS = 10
LR = 5e-4
T = 2.0
LAMBDA_KD = 10.0
REPLAY_PER_CLASS = 10
DATA_ROOT = "Data"
CONFIG_PATH = os.path.join(DATA_ROOT, "stage_config.json")

os.makedirs("model_pretrain", exist_ok=True)
os.makedirs("Record", exist_ok=True)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    stage_config = json.load(f)

prev_stage = f"{int(CURRENT_STAGE) - 1:02d}"
MODEL_PATH = os.path.join("model_pretrain", f"stage{prev_stage}_model.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

 
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

def get_cumulative_old_dataset(old_class_names, data_root, transform):
    class_to_idx = {cls: idx for idx, cls in enumerate(old_class_names)}
    samples = []
    train_path = os.path.join(data_root, "train")
    for cls in old_class_names:
        cls_path = os.path.join(train_path, cls)
        if os.path.isdir(cls_path):
            for fname in sorted(os.listdir(cls_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    path = os.path.join(cls_path, fname)
                    samples.append((path, class_to_idx[cls]))
    class CumulativeDataset(torch.utils.data.Dataset):
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
    return CumulativeDataset(samples, transform)

class IncrementalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y + self.label_offset

def build_herding_replay_dataset(old_dataset, k, model, device, old_classes):
    selected = []
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()
    
    for cls in range(old_classes):
        class_data = []
        class_orig_indices = []
        for idx in range(len(old_dataset)):
            img, label = old_dataset[idx]
            if label == cls:
                class_data.append(img)
                class_orig_indices.append(idx)
        n = len(class_data)
        if n <= k:
            selected.extend(class_orig_indices)
            continue
        feats_list = []
        bs = 32
        with torch.no_grad():
            for i in range(0, n, bs):
                batch = torch.stack(class_data[i:i+bs]).to(device)
                feat = feature_extractor(batch)
                feat = feat.view(batch.size(0), -1).cpu()
                feats_list.append(feat)
        feats = torch.cat(feats_list, dim=0)
        class_mean = feats.mean(dim=0)
        
        selected_cls = []
        remaining = list(range(n))
        sum_selected = torch.zeros_like(class_mean)
        for step in range(k):
            num_sel = len(selected_cls)
            target = ((num_sel + 1) * class_mean - sum_selected).unsqueeze(0)
            rem_feats = feats[remaining]
            dists = torch.norm(rem_feats - target, dim=1)
            best_local = dists.argmin().item()
            best_idx = remaining[best_local]
            selected_cls.append(best_idx)
            sum_selected += feats[best_idx]
            remaining.pop(best_local)
        for local in selected_cls:
            selected.append(class_orig_indices[local])
    return Subset(old_dataset, selected)

def evaluate(model, loader, old_class_indices):
    model.eval()
    correct = total = 0
    old_correct = old_total = 0
    new_correct = new_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            for i in range(len(y)):
                if y[i].item() in old_class_indices:
                    old_total += 1
                    if preds[i] == y[i]:
                        old_correct += 1
                else:
                    new_total += 1
                    if preds[i] == y[i]:
                        new_correct += 1
    return (correct/total if total > 0 else 0.0,
            old_correct/old_total if old_total > 0 else 0.0,
            new_correct/new_total if new_total > 0 else 0.0)

def expand_classifier(model, total_classes):
    old_fc = model.fc
    in_features = old_fc.in_features
    old_classes = old_fc.out_features
    new_fc = nn.Linear(in_features, total_classes)
    with torch.no_grad():
        new_fc.weight[:old_classes] = old_fc.weight
        new_fc.bias[:old_classes] = old_fc.bias
    model.fc = new_fc
    return model
 
def main():
    print(f"=== iCaRL Training (herding) - Stage {CURRENT_STAGE} ===")
    print(f"Loading previous model: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    old_classes = checkpoint['num_classes']
    old_class_names = checkpoint['class_names']

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, old_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
 
    all_learned_classes = []
    for s in range(1, int(CURRENT_STAGE) + 1):
        key = f"stage{s:02d}"
        if key in stage_config:
            all_learned_classes.extend(stage_config[key])

    new_classes = stage_config[f"stage{CURRENT_STAGE}"]
    print(f"New classes in stage {CURRENT_STAGE}: {new_classes}")
 
    old_dataset = get_cumulative_old_dataset(old_class_names, DATA_ROOT, transform)

   
    full_new_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
    
    new_class_to_idx = {cls: idx for idx, cls in enumerate(new_classes)}
    new_samples = []
    for path, label in full_new_dataset.samples:
        cls_name = full_new_dataset.classes[label]
        if cls_name in new_classes:
            new_label = new_class_to_idx[cls_name]  
            new_samples.append((path, new_label))

    class NewDataset(torch.utils.data.Dataset):
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

    new_dataset = NewDataset(new_samples, transform)
    print(f"Debug → 新類別樣本數: {len(new_dataset)} | 舊類別 replay: {REPLAY_PER_CLASS} × {old_classes} = {REPLAY_PER_CLASS * old_classes}")

    # Herding replay
    replay_dataset = build_herding_replay_dataset(old_dataset, REPLAY_PER_CLASS, teacher_model, device, old_classes)

 
    new_dataset_inc = IncrementalDataset(new_dataset, old_classes)
    combined_dataset = ConcatDataset([new_dataset_inc, replay_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Test loader
    current_class_names = all_learned_classes
    test_root = os.path.join(DATA_ROOT, "test")
    _, test_loader = get_test_loader(current_class_names, test_root, transform, BATCH_SIZE)

    old_class_indices = list(range(old_classes))
    total_classes = old_classes + len(new_classes)

    model = expand_classifier(model, total_classes).to(device)

    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': LR},
        {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': LR * 0.1}
    ])

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    records = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_ce = ce_loss(out, y)

            with torch.no_grad():
                teacher_out = teacher_model(x)

            # Knowledge Distillation  
            student_log_prob = nn.functional.log_softmax(out[:, :old_classes] / T, dim=1)
            teacher_prob = nn.functional.softmax(teacher_out[:, :old_classes] / T, dim=1)
            loss_kd = kl_loss(student_log_prob, teacher_prob) * (T * T)

            loss = loss_ce + LAMBDA_KD * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc, old_acc, new_acc = evaluate(model, test_loader, old_class_indices)
        records.append({
            'epoch': epoch + 1,
            'overall': round(acc, 4),
            'old': round(old_acc, 4),
            'new': round(new_acc, 4)
        })
        print(f"Epoch {epoch+1:2d} → Overall: {acc:.4f} | Old: {old_acc:.4f} | New: {new_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\nStage {CURRENT_STAGE} Training Time: {train_time:.2f} sec")

    record_path = os.path.join("Record", f"stage{CURRENT_STAGE}_record.csv")
    pd.DataFrame(records).to_csv(record_path, index=False)
    print(f"Stage {CURRENT_STAGE} record saved → {record_path}")

    SAVE_PATH = os.path.join("model_pretrain", f"stage{CURRENT_STAGE}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': total_classes,
        'class_names': current_class_names,
        'stage_acc': records[-1]['overall'],
        'old_acc': records[-1]['old'],
        'new_acc': records[-1]['new']
    }, SAVE_PATH)
    print(f"Stage {CURRENT_STAGE} model saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()