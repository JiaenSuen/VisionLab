import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from models.utils import check_accuracy,save_model,load_model
from models._modelRouter import modelRouter,Existing_model_names
from TrainingSection.dataset import datasetRouter_dict ,Existing_dataset_names,num_classes_dict
from tqdm import tqdm







def Train(modelName="", datasetName="",device="cuda",epochs=100):

    

    if modelName.lower() not in  Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return
    if datasetName.lower() not in  Existing_dataset_names:
        print(f"Dataset {datasetName} not recognized. Available datasets are: {Existing_dataset_names}")
        return
    


    # Training Params
    learning_rate = 1e-3
    batch_size = 1024
    num_epochs = epochs

    # Model
    model = modelRouter[modelName](num_classes=num_classes_dict[datasetName.lower()]) 
    model.to(device)

    if modelName == 'alexnet':
        input_size = 224
    else:
        input_size = 32

    # Data Loader
    train_loader = datasetRouter_dict[datasetName].GetTrainLoader(batch_size=batch_size ,input_size=input_size)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch_idx , (data,targets) in enumerate(pbar):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss   = criterion(scores,targets)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = sum(losses) / len(losses)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    #print(f"Cost at epoch {epoch+1} is {sum(losses)/len(losses):.5f}")
    final_avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {final_avg_loss:.4f}")

    check_accuracy(train_loader, model)
    save_model(model, f"trainedRelease/{datasetName}_{modelName}.pth")
    pass


