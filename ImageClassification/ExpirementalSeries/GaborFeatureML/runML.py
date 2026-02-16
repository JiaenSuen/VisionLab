import os
import cv2
import joblib   
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, freeze_support

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.filters import sobel

import xgboost  as xgb
import lightgbm as lgb
from sklearn.ensemble import (
    RandomForestClassifier, 
)




DATASET_DIR = "data"   
IMAGE_SIZE  = 128
MODEL_FILENAME = "gabor_model.pkl"
 


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



def process_image(args):
    img_path, class_name = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Read failed: {img_path}")
            return None, None
        feat = extract_features(img)   
        del img                  
        return feat, class_name
    except Exception as e:
        print(f"The error is in {img_path}: {e}")
        return None, None


def load_dataset(dataset_dir, max_workers=5):
    all_images = []
    
    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            all_images.append((img_path, class_folder))
    
    num_images = len(all_images)
    print(f"A total of {num_images} images were found, using {max_workers} processes.")
    
    if max_workers < 1:
        max_workers = 1
    
    results = []
    if num_images > 0:
        with Pool(processes=max_workers) as pool:
            results = list(tqdm(
                pool.imap(process_image, all_images),
                total=num_images,
                desc="Extract Features",
                unit="img"
            ))
    
    data = []
    labels = []
    for feat, label in results:
        if feat is not None:
            data.append(feat)
            labels.append(label)
 
    return np.array(data), np.array(labels)











def create_directory(base_dir, model_name):
    subdir = os.path.join(base_dir, model_name)
    os.makedirs(subdir, exist_ok=True)
    return subdir

def train_and_evaluate(model_name, clf, X_train, y_train, X_test, y_test, le, base_dir='Result'):
    print(f"training {model_name}...")
    subdir = create_directory(base_dir, model_name)
    

    clf.fit(X_train, y_train)
    
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    

    REPORT_FILENAME = os.path.join(subdir, f"{model_name}_report.txt")
    with open(REPORT_FILENAME, "w", encoding='utf-8') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Result Save : {REPORT_FILENAME}")
    

    fig = plt.figure(figsize=(14, 6))
    # Left：Confusion Matrix
    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap="Blues", cbar=False, ax=ax1)
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")
    ax1.set_title(f"Confusion Matrix - {model_name}")

    # Right：Classification Report 
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')  
    ax2.text(0, 1.0, f"Accuracy : {acc:.4f}", 
             fontsize=14, fontweight='bold', va='top')
    ax2.text(0, 0.9, "Classification Report:", 
             fontsize=13, fontweight='bold')
    ax2.text(0, 0.8, report, 
             fontfamily='monospace', fontsize=11.5,
             va='top', linespacing=1.4)

    plt.suptitle(f"Model Evaluation - {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()

    COMBINED_FILENAME = os.path.join(subdir, f"{model_name}_cm_report.png")
    plt.savefig(COMBINED_FILENAME, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined CM + Report saved: {COMBINED_FILENAME}")










if __name__ == '__main__':
    print("Data loading and feature extracting ...")
    freeze_support()
    X, y = load_dataset(DATASET_DIR)



    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # data for train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    
    print("training...")


    base_dir = 'Result'

 
    # LightGBM
    lgb_clf = lgb.LGBMClassifier(
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
    train_and_evaluate('lgb', lgb_clf, X_train, y_train, X_test, y_test, le, base_dir)

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    train_and_evaluate('xgb', xgb_clf, X_train, y_train, X_test, y_test, le, base_dir)

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    train_and_evaluate('rf', rf_clf, X_train, y_train, X_test, y_test, le, base_dir)
 

 
    print("All models trained and results saved.")