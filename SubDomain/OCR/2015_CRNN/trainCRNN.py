# train.py
import os
import numpy as np
from matplotlib import pyplot as plt

import torch
from utils import *
from crnn_model import CRNN
from dataset import CaptchaDataset , get_captcha_paths_and_labels , collate_fn 
from torch.utils.data import random_split, DataLoader


os.makedirs("model_saved",exist_ok=True)

NUM_EPOCHS = 500


def GetDataLoader(dataset_path = "dataset"): # return train/test loader
    image_paths, targets, orig_labels = get_captcha_paths_and_labels(dataset_path)
    dataset = CaptchaDataset(image_paths, targets, orig_labels, resize=(IMAGE_HEIGHT, IMAGE_WIDTH))


    GENERATOR = torch.Generator().manual_seed(42)  

    # Example of a split ratio : 70% training, 30% testing
    total = len(dataset)
    train_size = int(0.70 * total)
    test_size  = total - train_size

    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=GENERATOR 
    )

    print(f"Total Samples : {total}  | Train :  {len(train_ds)} | Test : {len(test_ds)} \n")
  


    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=1, pin_memory=True , collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=1, pin_memory=True , collate_fn=collate_fn)

    return train_loader , test_loader , dataset


def run_train(train_loader, test_loader, epochs=NUM_EPOCHS):
    
    print("Start Training")
    if os.path.exists(DEFAULT_MODEL_SAVE_PATH) :
        model = ModelOperator.build_load_RCNN(model_saved_path = DEFAULT_MODEL_SAVE_PATH)
    else : model = CRNN(num_chars=len(StringOperations.CHARS)).to(DEVICE)
 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True
    )

 
    best_epoch = 0
    best_accuracy = 0.0
    new_accuracy  = 0.0

    train_loss_record = []
    valid_loss_record = []

    for epoch in range(epochs):
 
        train_loss = ModelEngine.train_fn(model, train_loader, optimizer)

 
        all_preds, valid_loss = ModelEngine.eval_fn(model, test_loader)

 
        all_pred_strings = []
        all_true_strings = []

 
        for batch_preds in all_preds:
  
            for i in range(batch_preds.size(1)):  
                pred_seq = batch_preds[:, i, :]   # [T, C]
                pred_str = StringOperations.ctc_decode(pred_seq)
                all_pred_strings.append(pred_str)

        train_loss_record.append(train_loss)
        valid_loss_record.append(valid_loss)
        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Train Loss : {train_loss:.4f}  |  "
              f"Valid Loss : {valid_loss:.4f}")

        scheduler.step(valid_loss)

        if (epoch+1)%10 ==0:
            acc = evaluate_model(model, test_loader)
            new_accuracy = acc["char_accuracy"]

            if new_accuracy >= best_accuracy:   
                best_accuracy = new_accuracy
                best_epoch = epoch + 1
                torch.save(model.state_dict(), DEFAULT_MODEL_SAVE_PATH)
                print(f"-- Best model saved at epoch {best_epoch} (loss: {best_accuracy:.4f})")
        print("")

    print(f"\nTraining complete! The best model was achieved at epoch {best_epoch}, with accuracy = {best_accuracy:.2f}.")
    print(f"The model has been saved to : {DEFAULT_MODEL_SAVE_PATH}\n\n")

    
    test_result = evaluate_model(model, test_loader ,visualize=True)
    with open("record/test_result.txt", "w", encoding="utf-8") as f:
        f.write(str(test_result))

    train_result = evaluate_model(model, train_loader )
    with open("record/train_result.txt", "w", encoding="utf-8") as f:
        f.write(str(train_result))
    
    plot_training_curve(train_loss_record,valid_loss_record)

    return model



from Levenshtein import distance

def evaluate_model(model, test_loader , show=True , visualize=False):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    total_chars = 0
    total_edit_distance = 0

    total_samples = 0
    correct_sequences = 0

    sample_images = None
    sample_true_strings = []
    sample_pred_strings = []


    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Model"):
            images = batch["images"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)
            true_strings = batch["orig_label"]

            logits, loss = model(images, targets, lengths)
            if visualize and sample_images is None:
                sample_images = images.clone()

            total_loss += loss.item()
            num_batches += 1

            # logits shape: [T, B, C]
            for i in range(logits.size(1)):
                pred_seq = logits[:, i, :]   # [T, C]
                pred_str = StringOperations.ctc_decode(pred_seq)
                true_str = true_strings[i]

                total_samples += 1

                # --- Sequence Accuracy ---
                if pred_str == true_str:
                    correct_sequences += 1

                # --- Character Accuracy (Edit Distance based) ---
                ed = distance(pred_str, true_str)
                total_edit_distance += ed
                total_chars += len(true_str)
                
                if visualize and len(sample_pred_strings) < 10:
                    sample_true_strings.append(true_str)
                    sample_pred_strings.append(pred_str)

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # CER = total edit distance / total characters
    cer = total_edit_distance / total_chars if total_chars > 0 else 0.0

    # Character Accuracy
    char_accuracy = (1 - cer) * 100

    # Sequence Accuracy
    seq_accuracy = (correct_sequences / total_samples) * 100 if total_samples > 0 else 0.0
    if show:
        print("\n======= Evaluation Result =======")
        print(f"Loss                   : {avg_loss:.4f}")
        print(f"Character Error Rate   : {cer:.4f}")
        print(f"Character Accuracy     : {char_accuracy:.2f}%")
        print(f"OCR Full Accuracy      : {seq_accuracy:.2f}%")
        print("=================================\n")

    if visualize and sample_images is not None:
        visualize_crnn_predictions(
            sample_images,
            sample_true_strings,
            sample_pred_strings,
            max_images=10,
            save_path="record/crnn_test.png",
        )

    return {
        "loss": avg_loss,
        "cer": cer,
        "char_accuracy": char_accuracy,
        "sequence_accuracy": seq_accuracy
    }




if __name__ == "__main__":
    train_loader , test_loader , dataset = GetDataLoader()
    run_train(train_loader,test_loader)