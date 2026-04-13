import torch
import time
import csv
import os
from tqdm import tqdm   

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, save_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_name = save_name

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=1e-4
        )

    def train(self, epochs=10):
        history = []
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(epochs):
 
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()

            history.append([epoch, train_loss, val_loss, train_acc, val_acc])

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"results/{self.save_name}.pth")

           
            print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f} | Best Val Acc : {best_acc:.4f}")

        total_time = time.time() - start_time
        self._save_history(history)
        
        return best_acc, total_time

    def _train_epoch(self):
        self.model.train()
        total, correct, loss_sum = 0, 0, 0
 
        progress_bar = tqdm(self.train_loader, desc="Training Batch", leave=False)

        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

           
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        progress_bar.close()

        return loss_sum / len(self.train_loader), correct / total

    def _validate(self):
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0

        with torch.no_grad():
           
            for x, y in tqdm(self.val_loader, desc="Validating", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                loss_sum += loss.item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return loss_sum / len(self.val_loader), correct / total

    def _save_history(self, history):
        os.makedirs("Record", exist_ok=True)
        with open(f"Record/{self.save_name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
            writer.writerows(history)