import torch 
import torchvision.transforms as transforms

device = torch.device('cuda')
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 100

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