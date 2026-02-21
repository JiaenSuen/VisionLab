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

from models.ResNetsV1 import *
from models.utils import check_accuracy,save_model,load_model
from models._modelRouter import modelRouter,Existing_model_names
from TrainingSection.dataset import datasetRouter_dict ,Existing_dataset_names,num_classes_dict
from models._modelRouter import modelRouter

def Test(modelName="", datasetName="",device="cuda"):
    if modelName.lower() not in  Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return
    if datasetName.lower() not in  Existing_dataset_names:
        print(f"Dataset {datasetName} not recognized. Available datasets are: {Existing_dataset_names}")
        return
    
    
    if not os.path.exists(f"trainedRelease/{datasetName}_{modelName}.pth"):
        print(f"Trained model file trainedRelease/{datasetName}_{modelName}.pth not found. Please train the model first.")
        return
    
    

    model = modelRouter[modelName](num_classes=num_classes_dict[datasetName.lower()]) 
    model = load_model(model, f"trainedRelease/{datasetName}_{modelName}.pth")


    test_loader = datasetRouter_dict[datasetName].GetTestLoader(batch_size=256)
    check_accuracy(test_loader, model)