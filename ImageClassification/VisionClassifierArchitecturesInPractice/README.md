# **Vision Classifier Architectures in Practice : A From-Scratch Experimental Benchmark across Models and Datasets**
## **Overview**
This project is a systematic empirical study of image classification architectures, ranging from early CNN-based models such as VGG and ResNet to modern transformer and SSM-based designs.

Rather than aiming for state-of-the-art performance, the goal is to understand architectural evolution, inductive biases, and training behaviors through from-scratch PyTorch implementations and controlled experiments across multiple datasets.


## **Classifier Models**
### CNN
* VGG
* ResNet
* DenseNet
* Inception
* MobileNetV2
* ShuffleNet
* EfficientNet
* RegNet
* Xception
* HRNet

### Vision-Transformer
* ViT
* Swin
* MLP-Mixer

### CNN-Transformer
* ConvNeXt



### Mamba-based
* Vision Mamba / SSM
* Hybrid Conv + SSM


## Datasets 
* STL-10
* Caltech-101/Caltech-256
* CIFAR-10 / CIFAR-100
* Oxford 102 Flowers
* Stanford Dogs 
* CUB-200
* Imagenette
* Imagewoof
* MedMNIST


CIFAR-10: 60,000 32x32 color images in 10 classes (50,000 train, 10,000 test) .

The following models are included : <br>

| Model        | Train Accuracy | Test Accuracy | Epoches   |  GFLOPs  |  
| -----------  | -------------  | ------------- | ----------|------------- | 
| GoogLeNet    | 95.16%         | 68.29%        | 200       |25.38
| GoogLeNet(P) | 99.63%         | 82.87%        | 100       |31.31
| ResNet18(P)  | 99.49%         | 86.01%        | 100       |144.25
| ResNet34(P)  | 83.63%         | 79.50%        | 1         |298.87
| DenseNet121  | XX.XX%         | XX.XX%        | 0         |
