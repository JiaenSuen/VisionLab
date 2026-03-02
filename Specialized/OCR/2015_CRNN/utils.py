# utils.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List, Tuple, Union, Any

"""
Catalog
    (1) String/Prediction/Label Operations
    (2) Image Transform
    (3) Read One Image
    (4) Model Operations
    (5) Model Engine : For Train/Test
"""



# HyperParameters
DATA_DIR = "dataset"
BATCH_SIZE = 8
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 2
DEVICE = "cuda"
DEFAULT_MODEL_SAVE_PATH = "model_saved/CRNN.pt"

# String/Prediction/Label Operations
class StringOperations:
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
    IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

    @classmethod
    def encode_string(cls, label_str: str) -> List[int]:
        target = []
        for c in label_str.upper():
            if c in cls.CHAR2IDX:
                target.append(cls.CHAR2IDX[c])
            else:
                raise ValueError(f"The character '{c}' is not supported in label: {label_str}")
        return target

    @classmethod
    def decode_ids(cls, ids: Union[List[int], Tuple[int], Any]) -> str:
        if isinstance(ids, (list, tuple)):
            return ''.join(cls.IDX2CHAR.get(i, "?") for i in ids)

        try:
            ids_list = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            return ''.join(cls.IDX2CHAR.get(int(i), "?") for i in ids_list)
        except:
            return "???"

    @classmethod
    def decode_label(cls, label) -> str:
        if hasattr(label, 'cpu') and hasattr(label, 'numpy'):  # torch.Tensor
            label = label.cpu().numpy().tolist()
        return ''.join(cls.IDX2CHAR.get(idx, "") for idx in label)

    @classmethod
    def ctc_decode(cls, pred, blank: int = len(CHARS)):

        if isinstance(pred, torch.Tensor):
            if pred.ndim == 2:
                pred = pred.argmax(dim=-1)
            pred = pred.cpu().numpy()

        result = []
        prev = blank
        for p in pred:
            if p != blank and p != prev:
                result.append(p)
            prev = p

        return ''.join(cls.IDX2CHAR.get(int(i), "?") for i in result)

 
    @staticmethod
    def remove_duplicates(x: str) -> str:
        if len(x) < 2:
            return x
        fin = ""
        for j in x:
            if not fin or j != fin[-1]:
                fin += j
        return fin
    
    @staticmethod
    def PredictionTensor_to_String(Tensor):
        return Tensor[:, 0, :]




# Image Transform
TRANSFORM_FOR_IMAGE = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



# Read One Image
def Read_Image(image_path:str): # Give Path String -> torch format image
    image = Image.open(image_path).convert("RGB")
    image = TRANSFORM_FOR_IMAGE(image).unsqueeze(0).to(DEVICE)
    return image
    

from crnn_model import CRNN
class ModelOperator:
    
    # Build & Load RCNN from Saved Path
    def build_load_RCNN(model_saved_path = DEFAULT_MODEL_SAVE_PATH):
        # Load the trained model
        model = CRNN(len(StringOperations.CHARS))   
        model.load_state_dict(torch.load(model_saved_path, map_location=DEVICE ,weights_only=True))
        model.to(DEVICE)
        return model






from sklearn import preprocessing
# Prediction function : Give [ Model entity , one image path ] -> Predicted String Result
def predict_image(model,image_path):

    image = Read_Image(image_path)
    # Perform prediction
    with torch.no_grad():
        preds = model(image)
        if isinstance(preds, tuple):
            preds = preds[0]  # Ensure we use only the prediction output

 
    predictions_string = StringOperations.PredictionTensor_to_String(preds)
    prediction = StringOperations.ctc_decode(predictions_string)
    return prediction





# Train & Test Function
from tqdm import tqdm
class ModelEngine:

    @staticmethod
    def train_fn(model, data_loader, optimizer):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(data_loader, desc="Training   "):
   
            images = batch["images"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)


            optimizer.zero_grad()
            logits, loss = model(images, targets, lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN/Inf. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float('inf')


    @staticmethod
    def eval_fn(model, data_loader):
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating "):
                images  = batch["images"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)
                lengths = batch["lengths"].to(DEVICE)
                logits, loss = model(images, targets, lengths)

                total_loss += loss.item()
                num_batches += 1
                all_logits.append(logits.cpu())  

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return all_logits, avg_loss
    






# Visualization 

import os
import matplotlib.pyplot as plt

def plot_training_curve(train_loss_record_list, eval_loss_record_list, save_path="record/losses.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))


    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    epochs = range(1, len(train_loss_record_list) + 1)


    ax.plot(
        epochs,
        train_loss_record_list,
        color="#ff2b2b",
        linewidth=2.5,
        label="Train Loss"
    )

    ax.plot(
        epochs,
        eval_loss_record_list,
        color="white",
        linewidth=2.5,
        linestyle="--",
        label="Eval Loss"
    )


    ax.set_title("Training Curve", fontsize=16, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Epoch", fontsize=12, color="white")
    ax.set_ylabel("Loss", fontsize=12, color="white")


    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


    ax.grid(color="#aa0000", linestyle="--", linewidth=0.5, alpha=0.3)


    legend = ax.legend(frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color("white")


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Loss curve saved to {save_path}")

import math
def visualize_crnn_predictions(
    images,
    true_strings,
    pred_strings,
    max_images=10,
    save_path="record/crnn_test.png",
    dpi=300,
):
    """
    images       : Tensor [B, 3, H, W]
    true_strings : list[str]
    pred_strings : list[str]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    images = images.detach().cpu()

    num_images = min(len(images), max_images)
    num_cols = 5
    num_rows = math.ceil(num_images / num_cols)

    fig = plt.figure(figsize=(num_cols * 3, num_rows * 1.5))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    for i in range(num_images):
        img = images[i]

        img = img * 0.5 + 0.5

        img = img.permute(1, 2, 0).numpy()
        img = img.clip(0, 1)

        true_str = true_strings[i]
        pred_str = pred_strings[i]

        correct = (true_str == pred_str)

        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        title_color = "green" if correct else "red"
        ax.set_title(
            f"True: {true_str}\nPred: {pred_str}",
            fontsize=9,
            color=title_color,
            pad=6
        )

    fig.suptitle(
        "CRNN OCR Prediction Visualization",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)