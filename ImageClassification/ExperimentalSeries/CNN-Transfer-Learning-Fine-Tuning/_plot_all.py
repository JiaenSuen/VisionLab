import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import matplotlib as mpl
import seaborn as sns

def plot_all_curves(record_dir="Record"):
    files = sorted(glob.glob(f"{record_dir}/*.csv"))
    
    if not files:
        print(f"Warning: No CSV files found in {record_dir}")
        return

   
    plt.style.use("seaborn-v0_8-pastel")
    
   
    pastel_colors = [
        "#A8DADC", "#F4A261", "#E76F51", "#2A9D8F", 
        "#E9C46A", "#264653", "#6D597A", "#B5838D",
        "#81B29A", "#F2CC8F"
    ]
    
    mpl.rcParams.update({
        'figure.figsize': (13, 8),
        'lines.linewidth': 2.8,
        'lines.solid_capstyle': 'round',
        'axes.labelsize': 14,
        'axes.titlesize': 17,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10.5,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linestyle': '--'
    })

    plt.figure()

    for idx, f in enumerate(files):
        df = pd.read_csv(f)
        name = os.path.basename(f).replace(".csv", "").replace("_", " ")
        
        color = pastel_colors[idx % len(pastel_colors)]

        # Train - Solid line (slightly darker)
        plt.plot(
            df["epoch"], df["train_acc"],
            linestyle="-",
            color=color,
            alpha=0.95,
            linewidth=3.0,
            label=f"{name} (train)"
        )

        # Validation - Dashed line (slightly lighter)
        plt.plot(
            df["epoch"], df["val_acc"],
            linestyle="--",
            color=color,
            alpha=0.85,
            linewidth=2.6,
            label=f"{name} (val)"
        )

    plt.xlabel("Epoch", labelpad=12)
    plt.ylabel("Accuracy", labelpad=12)
    plt.title("Training vs Validation Accuracy Across Models", pad=20, fontweight='medium')

   
    plt.legend(
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor='lightgray',
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02)
    )

    plt.ylim(0.5, 1.02)                   
    plt.xlim(left=0)

    plt.tight_layout()

    
    os.makedirs("results", exist_ok=True)
    save_path = "results/all_models_comparisonl.png"
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Save : {save_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_all_curves()