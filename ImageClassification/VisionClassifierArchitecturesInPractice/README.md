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
* ConvNeXt

### Vision-Transformer
* ViT
* Swin
* MLP-Mixer

### LNN

### Mamba-based
* Vision Mamba / SSM
* Hybrid Conv + SSM

### **CIFAR-10 Dataset**
CIFAR-10: 60,000 32x32 color images in 10 classes (50,000 train, 10,000 test) .

The following models are included : <br>

| Model       | Train Accuracy | Test Accuracy | Epoches   |FLOPs|  
| ----------- | -------------  | ------------- | ----------|------------- | 
| ResNet18    | 98.84%         | 84.65%        | 100       |
| VGG16       | XX.XX%         | XX.XX%        | 0         |
| DenseNet121 | XX.XX%         | XX.XX%        | 0         |
