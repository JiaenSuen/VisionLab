import os
from ultralytics import YOLO
import yaml





def train_model(data_yaml: str, epochs: int = 50, batch: int = 16, imgsz: int = 640):

    model = YOLO('yolo11n.pt')  
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project='runs/train',
        name='tlprr_yolov11',
        exist_ok=True
    )
    return results


def evaluate_model(weights: str, data_yaml: str, imgsz: int = 640):
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        task='detect'
    )
    return metrics


def main():
    # config data
    data_yaml = 'tlprr/data.yaml'


    print("start training YOLOv11 nano...")
    train_results = train_model(
        data_yaml=data_yaml,
        epochs=10,
        batch=16,
        imgsz=640
    )
    
    print("Training Completed , Save to : runs/train/tlprr_yolov11")

    best_weights = os.path.join('runs', 'train', 'tlprr_yolov11', 'weights', 'best.pt')
    print("Evaluation and Calculate mAP...")
    eval_metrics = evaluate_model(
        weights=best_weights,
        data_yaml=data_yaml,
        imgsz=640
    )
    print("Test Result : ", eval_metrics)

if __name__ == '__main__':
    main()

