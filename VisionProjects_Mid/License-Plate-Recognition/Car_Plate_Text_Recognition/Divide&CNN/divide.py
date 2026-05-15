import cv2
import numpy as np

# 讀取車牌圖片（已裁切，只包含車牌區域）
image = cv2.imread('license_plate.jpg')

# 1. 灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 二值化（可以視情況改為 adaptiveThreshold）
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. 可選：形態學操作（消除雜訊、強化字元結構）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 4. 找輪廓
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 過濾輪廓，取得字元區塊
char_regions = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 篩選符合文字尺寸的區域（需依你實際的車牌字體尺寸調整）
    if 10 < w < 100 and 20 < h < 100:
        char_regions.append((x, y, w, h))

# 6. 對文字區塊按 X 排序（從左到右）
char_regions = sorted(char_regions, key=lambda x: x[0])

# 7. 裁切每個字元
for i, (x, y, w, h) in enumerate(char_regions):
    char_img = image[y:y+h, x:x+w]
    cv2.imwrite(f'char/char_{i}.png', char_img)
