from TrainingSection.train import Train
from TrainingSection.test  import Test


if __name__ == "__main__":
    print("Starting Training and Testing Process")
    print("-------------------------------------")
    print("Select Training or Testing by input 1 or 2 respectively.")
    choice = input("Enter 1 for Training, 2 for Testing: ")
    if choice == '1':
        modelName = input("Enter the model name (e.g., resnet18): ")
        datasetName = input("Enter the dataset name (e.g., cifar10): ")
        Train(modelName=modelName, datasetName=datasetName)
    elif choice == '2':
        modelName = input("Enter the model name (e.g., resnet18): ")
        datasetName = input("Enter the dataset name (e.g., cifar10): ")
        Test(modelName=modelName, datasetName=datasetName)
    else:
        print("Invalid choice. Please enter 1 for Training or 2 for Testing.")
 