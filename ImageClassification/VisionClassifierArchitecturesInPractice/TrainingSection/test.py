import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm


from models.resnets import *
from models.utils import check_accuracy,save_model,load_model

from TrainingSection.dataset import datasetRouter_dict ,Existing_dataset_names,num_classes_dict
from models.modelRouter import modelRouter

def Test(modelName="", datasetName="",device="cuda"):

    model = modelRouter(modelName=modelName,num_classes=num_classes_dict[datasetName.lower()]).to(device)
    model = load_model(model, f"trainedRelease/{modelName}.pth")

    test_loader = datasetRouter_dict[datasetName].GetTestLoader(batch_size=512)
    check_accuracy(test_loader, model)