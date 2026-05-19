import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from models.utils import check_accuracy,save_model,load_model
from models._modelRouter import modelRouter,Existing_model_names
from TrainingSection.dataset import Food101_224Dataset,NUM_CLASSES_OF_FOOD101,DATASET_NAME
from tqdm import tqdm







def Train(modelName="",device="cuda",epochs=100):

    torch.cuda.empty_cache()

    if modelName.lower() not in  Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return
 
    


    # Training Params
    learning_rate = 1e-3
    batch_size = 8
    num_epochs = epochs

    # Model
    model = modelRouter[modelName](num_classes=NUM_CLASSES_OF_FOOD101) 
    model.to(device)


    # Data Loader
    train_loader = Food101_224Dataset.GetTrainLoader(batch_size=batch_size)
    
    
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
    save_model(model, f"trainedRelease/{DATASET_NAME}_{modelName}.pth")
    pass


