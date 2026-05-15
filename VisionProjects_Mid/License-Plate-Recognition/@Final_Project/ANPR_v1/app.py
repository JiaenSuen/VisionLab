import cv2
import numpy as np
import pytesseract
import re
from ultralytics import YOLO
from PIL import Image



MODEL_PATH = "yolov11_ANPR.pt"
IMAGE_PATH = "test/10.jpg"
OUTPUT_PATH = f"result/result_{IMAGE_PATH[5:-4]}.jpg"
cascade_path = "haarcascade_ANPR.xml"
model_slect = 'yolo' # or haar
#model_slect = 'haar'
enhance = False


# 1. Object Detection: YOLO License Plate Detection
'''Load the YOLO model and detect license plates, then return a list of bounding boxes [x1, y1, x2, y2].'''
def detect_license_plates(yolo_model_path: str, image: np.ndarray):
   
    
   
    model = YOLO(yolo_model_path)
    results = model.predict(image, save=False, save_txt=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


# 1. Object detection: Haar Cascade
def detect_license_plates_haar(image: np.ndarray,
                              cascade_path: str = "haarcascade_ANPR.xml",
                              min_size: tuple = (90, 30),
                              max_size: tuple = (400, 150),
                              scale_factor: float = 1.05,
                              min_neighbors: int = 8) :
    """
    Use Haar Cascade to detect license plates, returning a list of bounding boxes [x1, y1, x2, y2].

    Parameters:
        image: BGR image
        cascade_path: HaarCascade XML path
        min_size: Minimum license plate size (w, h)
        max_size: Maximum license plate size (w, h)
        scale_factor: ScaleFactor of detectMultiScale
        min_neighbors: MinNeighbors of detectMultiScale

    Returns:
        [(x1, y1, x2, y2), ...]
    """
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    rects = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=max_size
    )
    
    boxes = []
    for (x, y, w, h) in rects:
        #Filter by aspect ratio and edge density
        aspect = w/float(h)
        roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi, 100, 200)
        edge_density = edges.sum()/(w*h)
        if 2.0 < aspect < 6.0 and edge_density > 0.1:
            boxes.append((x, y, x+w, y+h))
    return boxes


# 2. Perspective correction: Corrects tilted license plates to a horizontal viewing angle.
def rectify_plate(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:
        return img  # Skip calibration if no quadrilateral can be found
    # Sort by top left, top right, bottom right, bottom left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
    maxW, maxH = int(max(widthA, widthB)), int(max(heightA, heightB))

    dst = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    return warped



# 3. Image enhancement: Zoom in + Clarify + Sharpen
def enhance_image(img: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp


# 4. OCR identification: Tesseract + whitelist + post-processing
def ocr_plate(img: np.ndarray) -> str:
    pil = Image.fromarray(img)
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
    text = pytesseract.image_to_string(pil, config=config)
    return re.sub(r'[^A-Z0-9]', '', text)




# Main App Section :
def run_end_to_end(yolo_model_path: str, image_path: str, output_path: str):
    # 5.1 load
    orig = cv2.imread(image_path)
    display = orig.copy()

    # 5.2  detect
    if model_slect == "yolo" :
        boxes = detect_license_plates(yolo_model_path, orig)
    elif model_slect == "haar" :
        boxes = detect_license_plates_haar(orig, cascade_path)

    # 5.3 For each license plate box, perform correction, enhancement, and OCR.
    
    for (x1, y1, x2, y2) in boxes:
        roi = orig[y1:y2, x1:x2]
        captured_image = roi 
        if enhance :
            rect = rectify_plate(roi)
            enh  = enhance_image(rect, scale=2)
            captured_image  = enh
        
        text = ocr_plate(captured_image)


 
        cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(display, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 2)

    # 5.4 Store or display results
    cv2.imwrite(output_path, display)
    print(f"The results have been stored : {output_path}")



if __name__ == "__main__":
    run_end_to_end(MODEL_PATH, IMAGE_PATH, OUTPUT_PATH)
