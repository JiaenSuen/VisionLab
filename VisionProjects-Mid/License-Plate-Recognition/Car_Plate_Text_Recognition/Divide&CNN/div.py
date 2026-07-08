import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

 
image_path = "2.jpeg"
image = cv2.imread(image_path)

 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarization (using Otsu)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operations (closing operation to concatenate characters)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# Contour Detection
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering and sorting text outlines
char_regions = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 10 < w < 60 and 20 < h < 70:   
        char_regions.append((x, y, w, h))

# Sort by x (from left to right)
char_regions = sorted(char_regions, key=lambda b: b[0])

# Extract character images from the original image
char_images = [image[y:y+h, x:x+w] for (x, y, w, h) in char_regions]

 
char_pil_images = [Image.fromarray(cv2.cvtColor(char, cv2.COLOR_BGR2RGB)) for char in char_images]
char_pil_images[:6]  










'''


import cv2
import numpy as np
import os

def segment_chars_by_projection(img_path, 
                                 out_dir="chars", 
                                 thresh_block=11, 
                                 thresh_C=2, 
                                 min_char_width=10):
 
 
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, thresh_block, thresh_C)
 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    
 
    vertical_proj = np.sum(bw==255, axis=0)
    
 
    mean_val = np.mean(vertical_proj)
    is_low = vertical_proj < (mean_val*0.5)   
    splits = []
    in_gap = False
    for x, flag in enumerate(is_low):
        if flag and not in_gap:
     
            in_gap = True
            start = x
        elif not flag and in_gap:
        
            end = x-1
            splits.append((start+end)//2)
            in_gap = False
 
    if in_gap:
        end = len(is_low)-1
        splits.append((start+end)//2)
    
   
    boundaries = [0] + splits + [bw.shape[1]]
    
 
    os.makedirs(out_dir, exist_ok=True)
 
    char_paths = []
    for i in range(len(boundaries)-1):
        x1, x2 = boundaries[i], boundaries[i+1]
        w = x2 - x1
        if w < min_char_width: 
            continue   
        crop = img[:, x1:x2]
    
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bw_crop = cv2.threshold(gray_crop,0,255,
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        horiz_proj = np.sum(bw_crop==255, axis=1)
        ys = np.where(horiz_proj>0)[0]
        if ys.size:
            y1, y2 = ys[0], ys[-1]
            crop = crop[y1:y2+1, :]
    
        out_path = os.path.join(out_dir, f"char_{i}.png")
        cv2.imwrite(out_path, crop)
        char_paths.append(out_path)
    
    print(f"A total of {len(char_paths)} character images were cut out and saved in '{out_dir}'")
    return char_paths

if __name__ == "__main__":
   
    chars = segment_chars_by_projection(
        img_path="2.jpeg",
        out_dir="chars",
        thresh_block=15,
        thresh_C=3,
        min_char_width=15
    )

'''