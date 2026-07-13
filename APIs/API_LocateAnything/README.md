# LocateAnything Local Project

This project stores the NVIDIA LocateAnything model and Hugging Face cache
inside the project directory.

## Project structure

```text
LocateAnythingProject/
|-- download_model.py
|-- locate_anything_api.py
|-- locate_anything_image.py
|-- requirements.txt
|-- input.jpg
|-- models/
|   `-- LocateAnything-3B/
|-- outputs/
`-- .cache/
    `-- huggingface/
```

## Installation

Install a CUDA-enabled PyTorch build that matches the local CUDA environment,
then install the remaining packages:

```powershell
pip install -r requirements.txt
```

The two imports reported by the model are provided by these packages:

```powershell
pip install lmdb==1.7.5 decord==0.6.0
```

## Download the model

```powershell
python download_model.py
```

The download can be resumed and is saved under
`models/LocateAnything-3B/`.

## Run the single-image example

Place an image at `input.jpg`, edit `CATEGORIES` if needed, and run:

```powershell
python locate_anything_image.py
```

## Use the API in another project module

```python
from locate_anything_api import LocateAnythingAPI


locator = LocateAnythingAPI()
result = locator.detect(
    image="camera_frame.jpg",
    categories=["robot", "person", "workbench"],
    output_path="outputs/result.jpg",
)

for detection in result.detections:
    print(detection.label, detection.box)
```

Load `LocateAnythingAPI` only once and reuse the same instance for incoming
camera frames or server requests. Reloading the model for every image is slow
and wastes GPU memory.

The official model currently lists Linux as its supported operating system.
On Windows, the standard PyTorch SDPA fallback may work without MagiAttention.
If hybrid decoding causes a Windows-specific error, set
`generation_mode="slow"` as the compatibility fallback.
