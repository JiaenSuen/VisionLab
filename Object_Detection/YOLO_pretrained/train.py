from ultralytics import YOLO
import shutil

if __name__ == "__main__":
    model = YOLO("weights/yolo11n.pt")

    results = model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=512,
        batch=4,
        workers=2  
    )

    shutil.copy("runs/detect/train/weights/best.pt", "weights/best.pt")
    shutil.copy("runs/detect/train/weights/last.pt", "weights/last.pt")
