import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


'''
#Cifar-10 normalization values:
transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
'''



class Cifar10Dataset:
    default_train_transform = transforms.Compose([transforms.ToTensor(),])
    default_test_transform = transforms.Compose([transforms.ToTensor(),])

    @staticmethod
    def GetTrainLoader(batch_size,transformFunc=None):
        if transformFunc is None:
            transformFunc = Cifar10Dataset.default_train_transform
        train_dataset = torchvision.datasets.CIFAR10(
            root="dataset/", train=True, transform=transformFunc, download=True
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader


    @staticmethod
    def GetTestLoader(batch_size,transformFunc=None):
        if transformFunc is None:
            transformFunc = Cifar10Dataset.default_test_transform
        test_dataset = torchvision.datasets.CIFAR10(
            root="dataset/", train=False, transform=transformFunc, download=True 
        )
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader



num_classes_dict = {
    "cifar10": 10,
    "cifar100": 100,
}



datasetRouter_dict = {
    "cifar10": Cifar10Dataset,

}
Existing_dataset_names = list(datasetRouter_dict.keys())