import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Food-101 224x224

NUM_CLASSES_OF_FOOD101 = 101
class Food101_224Dataset:
    """
    Food-101 224x224 dataset.

    Classes: 101
    Train split: train
    Test split: test

    Recommended for:
    - from-scratch training
    - architecture comparison
    - texture / deformation robustness
    """

    default_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    default_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    @staticmethod
    def GetTrainLoader(batch_size, input_size=224, transformFunc=None, augment=True):
        if transformFunc is None:
            if augment:
                transformFunc = transforms.Compose([
                    transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            else:
                transformFunc = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])

        train_dataset = torchvision.datasets.Food101(
            root="dataset/",
            split="train",
            transform=transformFunc,
            download=True,
        )

        train_dataset.train = True

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader

    @staticmethod
    def GetTestLoader(batch_size, input_size=224, transformFunc=None):
        if transformFunc is None:
            transformFunc = transforms.Compose([
                transforms.Resize(int(input_size * 256 / 224)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        test_dataset = torchvision.datasets.Food101(
            root="dataset/",
            split="test",
            transform=transformFunc,
            download=True,
        )

        test_dataset.train = False

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return test_loader

 

