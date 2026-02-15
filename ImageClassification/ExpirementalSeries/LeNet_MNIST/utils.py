import math
import matplotlib.pyplot as plt
import torch

def plot_img(image,label,normalized=True):
    image = image.numpy()[0]
    if normalized:
        mean = 0.1307
        std  = 0.3081
        image = (image * std) + mean
    plt.title(label=label.item())
    plt.imshow(image,cmap='gray')
    plt.show()




def visualize_predictions(
    images,
    labels,
    predictions,
    mean=None,
    std=None,
    max_images=10,
    show=True,
    save=True,
    save_path="mnist_predictions.png",
    dpi=300
):
    """
        images: Tensor [B, 1, H, W]
        labels: Tensor [B]
        predictions: Tensor [B] (can contain -1 or None-like for no prediction)
        mean/std: optional normalization parameters
        max_images: max number of images to display
        show: whether to display figure
        save: whether to save figure
        save_path: output filename
        dpi: save resolution
    """

    images = images.detach().cpu()
    labels = labels.detach().cpu()
    predictions = predictions.detach().cpu()

    num_images = min(len(images), max_images)
    num_cols = 5
    num_rows = math.ceil(num_images / num_cols)

    fig = plt.figure(figsize=(num_cols * 2.2, num_rows * 2.2))
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    for i in range(num_images):
        img = images[i][0].numpy()

        # Restore if normalized
        if mean is not None and std is not None:
            img = img * std + mean

        label = labels[i].item()

        pred_raw = predictions[i].item()
        pred = "X" if pred_raw == -1 else pred_raw

        correct = (pred_raw == label)

        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        # Use color to distinguish between "Correct" and "Incorrect"
        title_color = "green" if correct else "red"
        ax.set_title(
            f"Label: {label}\nPred: {pred}",
            fontsize=10,
            color=title_color,
            pad=10
        )


    fig.suptitle(
        "LeNet on MNIST - Prediction Visualization",
        fontsize=16,
        fontweight="bold",
        y=1.1
    )

    if save:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:    plt.show()
    else:       plt.close(fig)
 





def plot_training_curve(train_losses, train_accs, test_losses, test_accs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(train_losses, label="Train")
    ax[0].plot(test_losses, label="Test")
    ax[0].set_title("Loss Curve")
    ax[0].legend()

    ax[1].plot(train_accs, label="Train")
    ax[1].plot(test_accs, label="Test")
    ax[1].set_title("Accuracy Curve")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("Record/training_curve.png", dpi=300)



import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
def plot_confusion_matrix(all_labels_list, all_preds_list, num_classes=None, class_names=None, save_path = "Record/ConfusionMatrix.png"):
    """
    Convert the collected list into a confusion matrix and display a heatmap.
    """
    all_labels = torch.cat(all_labels_list).numpy()
    all_preds  = torch.cat(all_preds_list).numpy()

    cm = confusion_matrix(all_labels, all_preds)

    # Automatically infer the number of categories
    if num_classes is None:num_classes = len(np.unique(all_labels))
    if class_names is None:class_names = [f"Class {i}" for i in range(num_classes)]

    print("\n Classification Report :")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)

    return cm
