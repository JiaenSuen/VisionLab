import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from models.utils import check_accuracy,save_model,load_model
from models.modelRouter import modelRouter,Existing_model_names
from TrainingSection.dataset import datasetRouter_dict ,Existing_dataset_names,num_classes_dict
from tqdm import tqdm







def Train(modelName="", datasetName="",device="cuda"):

    

    if modelName.lower() not in  Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return
    if datasetName.lower() not in  Existing_dataset_names:
        print(f"Dataset {datasetName} not recognized. Available datasets are: {Existing_dataset_names}")
        return
    


    # Training Params
    learning_rate = 1e-3
    batch_size = 1024
    num_epochs = 1

    # Model
    model = modelRouter(modelName=modelName,num_classes=num_classes_dict[datasetName.lower()]) 
    model.to(device)

    # Data Loader
    train_loader = datasetRouter_dict[datasetName].GetTrainLoader(batch_size=batch_size)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        losses = []

        for batch_idx , (data,targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss   = criterion(scores,targets)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Cost at epoch {epoch+1} is {sum(losses)/len(losses):.5f}")


    check_accuracy(train_loader, model)
    save_model(model, f"trained/{datasetName}_{modelName}.pth")
    pass


