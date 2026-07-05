# visualize_records.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.figsize': (12, 7),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

record_dir = "Record"
csv_files = sorted(
    [f for f in os.listdir(record_dir) if f.startswith("stage") and f.endswith("_record.csv")],
    key=lambda x: int(x[5:7])                     # stage01 → 01, stage02 → 02 ...
)

if not csv_files:
    print("❌ Record 資料夾內沒有找到任何 stageXX_record.csv")
    exit()

print(f"找到 {len(csv_files)} 個 stage 紀錄，正在合併並繪製論文級圖表...")

all_data = []
global_epoch_offset = 0
for csv_file in csv_files:
    stage_num = int(csv_file[5:7])
    df = pd.read_csv(os.path.join(record_dir, csv_file))
    df['stage'] = stage_num
    df['global_epoch'] = df['epoch'] + global_epoch_offset
    all_data.append(df)
    global_epoch_offset += len(df)                # 下一階段接續

full_df = pd.concat(all_data, ignore_index=True)

# 轉成長格式方便 seaborn 繪圖
melted = full_df.melt(
    id_vars=['global_epoch', 'stage', 'epoch'],
    value_vars=['overall', 'old', 'new'],
    var_name='Metric',
    value_name='Accuracy'
)
melted['Metric'] = melted['Metric'].map({
    'overall': 'Overall Accuracy',
    'old': 'Old Classes Accuracy',
    'new': 'New Classes Accuracy'
})

# 開始繪製
fig, ax = plt.subplots()
sns.lineplot(
    data=melted,
    x='global_epoch',
    y='Accuracy',
    hue='Metric',
    style='Metric',
    markers=True,
    dashes=False,
    linewidth=2.5,
    markersize=6,
    palette=['#1f77b4', '#ff7f0e', '#2ca02c']
)

# 加上每個 stage 的分隔線（虛線）
for i in range(1, len(csv_files) + 1):
    epoch_boundary = i * 10
    ax.axvline(x=epoch_boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1)

# 美化
ax.set_xlabel("Training Epoch (Stage 1 → Stage N)", fontsize=14)
ax.set_ylabel("Accuracy", fontsize=14)
ax.set_title("Repertorium Oecologicum\n"
             "Scalable Long-Tail Class-Incremental Learning Accuracy Curves\n"
             "(Old / New / Overall per Epoch)", pad=20)
ax.grid(True, linestyle='--', alpha=0.5)

# Legend 放在右上
ax.legend(title="Metric", loc='lower right', frameon=True)

# 在每個 stage 開始位置標註 Stage 編號
for i, csv_file in enumerate(csv_files):
    stage_num = int(csv_file[5:7])
    x_pos = i * 10 + 0.5
    ax.text(x_pos, 0.02, f"Stage {stage_num}", 
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=11, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(record_dir, "incremental_accuracy_curves.pdf"), dpi=300)
plt.savefig(os.path.join(record_dir, "incremental_accuracy_curves.png"), dpi=300)
 

print("✅ 論文級視覺化完成！")
print("   → Record/incremental_accuracy_curves.pdf")
print("   → Record/incremental_accuracy_curves.png")