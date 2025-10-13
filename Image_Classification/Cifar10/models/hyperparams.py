import torch 
import torchvision.transforms as transforms

device = torch.device('cuda')
num_classes = 10



train_transform = transforms.Compose([
    transforms.ToTensor(),
    
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

'''
transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
'''