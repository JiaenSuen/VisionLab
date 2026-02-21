import cv2
import numpy as np
from skimage.filters import sobel
from scipy.stats import skew, kurtosis

def extract_features(image, IMAGE_SIZE = 128):
    features = []
    # Preprocessing
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Histogram Equalization for Contrast Enhancement

    gamma = 0.5
    psi = 0
    # Gabor filters
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 2, 3]:
            for lamda in [np.pi/4, np.pi/2]:
                kernel = cv2.getGaborKernel((9, 9), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                # Statistical Features
                features.append(filtered.mean())              # Mean Average
                features.append(filtered.std())               # Standard Deviation
                features.append(np.sum(filtered ** 2))        # Energy
                features.append(skew(filtered.ravel()))       # Skewness
                features.append(kurtosis(filtered.ravel()))   # Kurtosis

    # Sobel edge
    sobel_img = sobel(gray)
    features.append(sobel_img.mean())
    features.append(sobel_img.std())
    features.append(np.sum(sobel_img ** 2))

    return np.array(features, dtype=np.float32)
