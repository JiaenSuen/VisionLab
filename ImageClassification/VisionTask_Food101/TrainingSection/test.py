import os
import torch 

from models.utils import check_accuracy, load_model
from models._modelRouter import modelRouter, Existing_model_names
from TrainingSection.dataset import Food101_224Dataset, NUM_CLASSES_OF_FOOD101, DATASET_NAME
from TrainingSection.training_recipe import get_training_config


def Test(modelName="", device="cuda"):
    modelName = modelName.lower()

    if modelName not in Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model_path = f"trainedRelease/{DATASET_NAME}_{modelName}_best.pth"

    if not os.path.exists(model_path):
        print(f"Best trained model file {model_path} not found. Please train the model first.")
        return

    config = get_training_config(modelName, epochs=1)

    model = modelRouter[modelName](num_classes=NUM_CLASSES_OF_FOOD101)
    model = load_model(model, model_path, device=device)

    EVAL_TRAIN_ACC = True

    if EVAL_TRAIN_ACC:
        train_loader = Food101_224Dataset.GetTrainLoader(
            batch_size=config["test_batch_size"],
            augment=False,
        )

        train_acc = check_accuracy(
            train_loader,
            model,
            calculate_profile=False
        )

        print(f"Train Accuracy of best checkpoint: {train_acc:.2f}%")

    test_loader = Food101_224Dataset.GetTestLoader(
        batch_size=config["test_batch_size"]
    )

    test_acc = check_accuracy(
        test_loader,
        model,
        calculate_profile=True
    )

    print(f"Test Accuracy of best checkpoint: {test_acc:.2f}%")