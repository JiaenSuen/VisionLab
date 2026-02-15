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
    default_test_transform  = transforms.Compose([transforms.ToTensor(),])

    @staticmethod
    def GetTrainLoader(batch_size, input_size=32, transformFunc=None, augment=False):
        if transformFunc is None:
            base_transform = [
                transforms.Resize((input_size, input_size)) if input_size != 32 else transforms.Lambda(lambda x: x),  
                transforms.ToTensor(),
            ]
            if augment:
                base_transform.insert(0, transforms.RandomCrop(32, padding=4))  
                base_transform.insert(1, transforms.RandomHorizontalFlip())
            transformFunc = transforms.Compose(base_transform)
        
        train_dataset = torchvision.datasets.CIFAR10(
            root="dataset/", train=True, transform=transformFunc, download=True
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    @staticmethod
    def GetTestLoader(batch_size, input_size=32, transformFunc=None):
        if transformFunc is None:
            base_transform = [
                transforms.Resize((input_size, input_size)) if input_size != 32 else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
            transformFunc = transforms.Compose(base_transform)
        
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