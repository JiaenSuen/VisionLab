from PIL import Image
import pytesseract
print(pytesseract.image_to_string(Image.open('test/license_plate.jpg')))




import cv2
import numpy as np
import pytesseract
import re
from PIL import Image

# 1.Image enhancement
def enhance_image(img, scale=2):
    # 1.1 enlarge
    h, w = img.shape[:2]
    img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # 1.2 CLAHE 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # 1.3  Unsharp Mask 
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp

# 2. Strabismus Perspective Correction
def rectify_plate(img):
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
        return img 
    
    rect = np.zeros((4,2), dtype="float32")# Sort by top left, top right, bottom right, bottom left
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
 
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # TTarget Coordinate 
    dst = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    return warped

#  OCR Prediction
def ocr_plate(img):
    pil = Image.fromarray(img)  
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7' #   A–Z、0–9
    text = pytesseract.image_to_string(pil, config=config)
    text = re.sub(r'[^A-Z0-9]', '', text)  
    return text

if __name__ == "__main__":
    img = cv2.imread("test/license_plate.jpg") 
    plate       = rectify_plate(img)
    enhanced    = enhance_image(plate, scale=2)
    result      = ocr_plate(enhanced)
    print("Identify strings : ", result)
