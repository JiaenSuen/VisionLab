from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

def get_dataloader(path, batch_size=32, augment=False):
    dataset = datasets.ImageFolder(path, transform=get_transforms(augment))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)