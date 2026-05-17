import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import os

from models.ShortcutSeries.ResNet import *
from models.utils import check_accuracy,save_model,load_model
from models._modelRouter import modelRouter,Existing_model_names
from TrainingSection.dataset import Food101_224Dataset,NUM_CLASSES_OF_FOOD101
from models._modelRouter import modelRouter

def Test(modelName="",device="cuda"):
    if modelName.lower() not in  Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return
 
    
    
    if not os.path.exists(f"trainedRelease/Food101_{modelName}.pth"):
        print(f"Trained model file trainedRelease/Food101_{modelName}.pth not found. Please train the model first.")
        return
    
    

    model = modelRouter[modelName](num_classes=NUM_CLASSES_OF_FOOD101) 
    model = load_model(model, f"trainedRelease/Food101_{modelName}.pth")


    test_loader = Food101_224Dataset.GetTestLoader(batch_size=256)
    check_accuracy(test_loader, model)