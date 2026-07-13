# Grounding DINO Tiny Local API

這是一個適合 Windows、PyTorch 與 Hugging Face 的 Grounding DINO Tiny 專案。權重會保存在專案內，API 載入一次後可持續接收圖片、PIL Image 或 OpenCV NumPy frame。

模型：`IDEA-Research/grounding-dino-tiny`（約 0.2B 參數，模型儲存庫約 1.4 GB）。

## 專案結構

```text
GroundingDINOTinyProject/
├─ download_model.py
├─ grounding_dino_api.py
├─ grounding_dino_image.py
├─ requirements.txt
├─ models/
│  └─ grounding-dino-tiny/
└─ outputs/
```

## Windows 安裝

建議 Python 3.10 或 3.11。若需要 NVIDIA GPU，先從 PyTorch 官網選擇與 CUDA 相符的安裝命令，再安裝其餘依賴：

```powershell
python -m pip install -r requirements.txt
```

下載模型至專案內：

```powershell
python download_model.py
```

## 單張圖片

先在 `grounding_dino_image.py` 頂部修改：

```python
IMAGE_PATH = Path("input.jpg")
OUTPUT_PATH = Path("outputs/result.jpg")
CATEGORIES = ["person", "car", "bicycle"]
```

然後直接執行：

```powershell
python grounding_dino_image.py
```

輸出預設保存在 `outputs/result.jpg`。

## 模組化 API

```python
from grounding_dino_api import GroundingDINOAPI, GroundingDINOConfig


config = GroundingDINOConfig(
    dtype="float32",
    box_threshold=0.35,
    text_threshold=0.25,
    input_short_side=640,
    input_long_side=960,
    nms_iou_threshold=0.60,
    connect_corner_edges=True,
    dash_length=10,
    dash_gap=7,
    dash_width=1,
    dash_lighten_ratio=0.55,
)

detector = GroundingDINOAPI(config)

result = detector.detect(
    image="input.jpg",
    categories=["person", "car", "bicycle"],
    output_path="outputs/result.jpg",
)

for detection in result.detections:
    print(detection.label, detection.score, detection.box)
```

自然語言定位：

```python
result = detector.ground(
    image="scene.jpg",
    description="the person wearing a red shirt",
    output_path="outputs/person.jpg",
)
```

OpenCV 相機畫面（API 預設 NumPy 為 BGR）：

```python
success, frame = camera.read()
if success:
    result = detector.detect(frame, ["person", "robot"])
```

雙目校正後若有無效黑邊，可只讓模型處理有效 ROI；輸出座標會自動映射回原圖：

```python
result = detector.detect(
    image="rectified_left.jpg",
    categories=["person", "robot"],
    valid_roi=(40, 20, 1240, 700),
)
```

## 嵌入式調整建議

- 平衡模式：`input_short_side=640`, `input_long_side=960`
- 更快模式：`input_short_side=480`, `input_long_side=736`
- 優先小物件召回：`input_short_side=800`, `input_long_side=1333`
- 誤偵測偏多：提高 `box_threshold` 至 `0.40` 或 `0.45`
- 漏偵測偏多：降低 `box_threshold` 至 `0.25` 或 `0.30`
- CPU 與 GPU 都預設使用 FP32；CUDA 會開啟 TF32 加速。不要將完整 Grounding DINO 模型直接轉成 FP16，部分 Transformers 版本的文字特徵仍保持 FP32，會造成 Float/Half dtype mismatch
- `compile_model=True` 可實驗 PyTorch 編譯，但 Windows／嵌入式相容性依平台而異，預設關閉

`result` 會提供：

- `result.detections`：label、score 與原圖像素座標
- `result.annotated_image`：PIL Image
- `result.prompt`：實際文字提示
- `result.elapsed_ms`：單次推論及後處理時間
- `result.as_dict()`：便於 Flask 或 JSON API 使用的資料

四角框預設會使用虛線連接四條邊。若只需要原本的四角框，可設定
`connect_corner_edges=False`；`dash_length` 與 `dash_gap` 分別控制虛線長度與間隔，
`dash_width` 控制虛線粗細，`dash_lighten_ratio` 控制向白色淡化的程度。

Grounding DINO Tiny 雖是 Tiny 版本，仍是約 0.2B 參數的開放詞彙 Transformer detector；它比 3B 級生成式定位模型輕很多，但不等同 YOLO Nano 等級。若目標是 Jetson Orin Nano 即時串流，後續仍建議評估 TensorRT/ONNX 或以固定類別 detector 作為快速路徑。
