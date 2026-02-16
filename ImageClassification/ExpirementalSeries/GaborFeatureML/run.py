import os
import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.filters import sobel
import joblib   
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
 
DATASET_DIR = "data"   
IMAGE_SIZE = 128
MODEL_FILENAME = "gabor_model.pkl"
REPORT_FILENAME = "classification_report.txt"


from scipy.stats import skew, kurtosis
def extract_features(image):
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

 
def load_dataset(dataset_dir):
    data = []
    labels = []
    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    feat = extract_features(img)
                    data.append(feat)
                    labels.append(class_folder)
    return np.array(data), np.array(labels)



print("data loading and feature extracting...")
X, y = load_dataset(DATASET_DIR)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# data for train / test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

 
print("training...")
#clf = RandomForestClassifier(n_estimators=250, random_state=42)
clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    force_col_wise=True
)

clf.fit(X_train, y_train)

 
#joblib.dump(clf, MODEL_FILENAME);print(f"model save : {MODEL_FILENAME}")

 
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

# Save
with open(REPORT_FILENAME, "w", encoding='utf-8') as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)
print(f"Result Save : {REPORT_FILENAME}")

 
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion Matrix Save : confusion_matrix.png")
