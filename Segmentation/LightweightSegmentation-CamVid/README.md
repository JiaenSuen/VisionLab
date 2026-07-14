# LightweightSegmentation

以 CamVid 11 類別資料集微調 `LR-ASPP + MobileNetV3-Large` 的第一版語意分割專案。

## 1. 安裝

建議使用 Python 3.10～3.12 與 CUDA 版 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

請依你的 CUDA／PyTorch 環境調整官方安裝指令，不必重複安裝已存在的 PyTorch。

## 2. 準備資料

將 CamVid 放成以下結構。標籤可與影像同名，也可使用 CamVid 常見的
`_L` 後綴，例如 `0001TP_009210.png` 對應 `0001TP_009210_L.png`：

```text
data/CamVid/
├── train/          # 影像
├── train_labels/   # mask
├── val/
├── val_labels/
├── test/
└── test_labels/
```

Dataset 支援常見的 CamVid RGB palette mask，以及像素值為 `0～10` 的灰階 class-ID mask。未知顏色會轉成 `ignore_index=255`。

先檢查資料與 mask：

```bash
python -m datasets.camvid --root data/CamVid --split train
```

## 3. 訓練

```bash
python train.py --config configs/lraspp.yaml
```

若只想做快速 smoke test：

```bash
python train.py --config configs/lraspp.yaml --epochs 1 --max-train-batches 2 --max-val-batches 2
```

## 4. 評估、預測與效能測試

```bash
python evaluate.py --config configs/lraspp.yaml \
  --checkpoint outputs/lraspp_experiment/checkpoints/best.pt \
  --save-predictions

python predict.py --config configs/lraspp.yaml \
  --checkpoint outputs/lraspp_experiment/checkpoints/best.pt \
  --image example.png

python benchmark.py --config configs/lraspp.yaml \
  --checkpoint outputs/lraspp_experiment/checkpoints/best.pt
```

## 5. 第一次執行時應確認

1. `python -m datasets.camvid` 顯示的 mask ID 僅包含 `0～10` 與可能的 `255`。
2. 先嘗試讓模型 overfit 少量 batch，再進行完整訓練。
3. 不要用 bilinear resize mask；本專案固定使用 nearest-neighbor。
4. 比較模型時固定影像尺寸、資料切分、loss 與測速方式。
