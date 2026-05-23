from TrainingSection.train import Train
from TrainingSection.test  import Test
from models._modelRouter import Existing_model_names


if __name__ == "__main__":
    2
    print("Starting Training and Testing Process")
    print("-------------------------------------")
    print(f"Available Models : {Existing_model_names}")
    print("Select Training or Testing by input 1 or 2 respectively.")
    choice = input("Enter 1 for Training, 2 for Testing : ")
    if choice == '1': 
        epochs_input = input("Set Epochs for Training (default is 100) : ")
        epochs = int(epochs_input) if epochs_input.isdigit() else 100

        modelName = input("Enter the model name (e.g., resnet18) : ")

        Train(modelName=modelName, epochs=epochs)
    elif choice == '2':
        modelName = input("Enter the model name (e.g., resnet18) : ")

        Test(modelName=modelName)
    else:
        print("Invalid choice. Please enter 1 for Training or 2 for Testing.")
 