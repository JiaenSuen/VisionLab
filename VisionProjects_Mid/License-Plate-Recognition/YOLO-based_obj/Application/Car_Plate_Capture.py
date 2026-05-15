import cv2
import os
from ultralytics import YOLO
from pathlib import Path

def predict_and_crop(
    model_path: str,
    image_path: str,
    save_dir: str = 'cropped_results'
):
    
    model = YOLO(model_path)    # Load YOLO Model

    # prediction
    results = model.predict(image_path, save=True, save_txt=False)

    
    os.makedirs(save_dir, exist_ok=True)
    image = cv2.imread(image_path)
    for i, r in enumerate(results):
        boxes = r.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # x1, y1, x2, y2 
            cropped = image[y1:y2, x1:x2]
            # Save ->
            base = Path(image_path).stem
            save_path = os.path.join(save_dir, f'{base}_crop{j+1}.jpg')
            cv2.imwrite(save_path, cropped)
            print(f'Save Capture : {save_path}')

if __name__ == '__main__':
    
    model_path = 'best.pt'
    image_path = 'test_image/03.jpg'

    predict_and_crop(model_path, image_path)
