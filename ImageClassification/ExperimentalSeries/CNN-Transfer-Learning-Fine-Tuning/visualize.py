import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_curve(csv_path):
    df = pd.read_csv(csv_path)
    name = os.path.basename(csv_path).replace(".csv","")

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train")
    plt.plot(df["epoch"], df["val_acc"], label="val")
    plt.title(name)
    plt.legend()
    plt.savefig(f"Record/{name}.png")

 