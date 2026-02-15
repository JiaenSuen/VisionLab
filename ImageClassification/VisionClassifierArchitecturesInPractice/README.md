# **Vision Classifier Architectures in Practice : A From-Scratch Experimental Benchmark across Models and Datasets**
## **Overview**
This project is a systematic empirical study of image classification architectures, ranging from early CNN-based models such as VGG and ResNet to modern transformer and SSM-based designs.

Rather than aiming for state-of-the-art performance, the goal is to understand architectural evolution, inductive biases, and training behaviors through from-scratch PyTorch implementations and controlled experiments across multiple datasets.


## **Classifier Models**
### CNN

| 2014 | 2015 |2016 |2017 |2018 |2019|2020|
| ---| ---  | --- | --- |--- |--- |--- |
|VGG|Highway|Xception|DenseNet|MobileNetV2|EfficientNet|RegNet|
|NIN|ResNet|SqueezeNet |ShuffleNet|ShuffleNetV2|HRNet|.|
|All-ConvNet|InceptionV2|Wide ResNet|MobileNet|MnasNet|.|.|
|InceptionV1|InceptionV3|ResNeXt|NASNet|.|.|.|
|.|.|ResNetV2|SENet |.|.|.|
|.|.|InceptionV4|.|.|.|.|

### Vision-Transformer /  CNN-Transformer
| 2020 | 2021 |2022 |2023 |2024 |2025|
| ---| ---  | --- | --- |--- |--- |
|ViT|Swin|ConvNeXt|ConvNeXt V2|MobileViT|Hybrid ViT-CNN|
|MLP-Mixer|EfficientNetV2|DaViT|DINOv2|EfficientViT|DINOv3
 

 

### Mamba-based
* Vision Mamba / SSM
* MambaVision
* Hybrid Conv + SSM
* Res-VMamba
* C-RADIOv4


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
 

### CIFAR-10
CIFAR-10: 60,000 32x32 color images in 10 classes (50,000 train, 10,000 test) .
Record Accuracy , Epochs , *GFLOPs Per Image*  <br><br>
The following models are included : <br>

| Model          | Train Accuracy | Test Accuracy | Epoches   |  GFLOPs  |  
| -------------  | -------------  | ------------- | ----------|------------- | 
| AlexNet (P)  | 00.00%         | 00.00%        | 100       |1.4203
| GoogLeNet    | 95.16%         | 68.29%        | 200       |0.0496
| GoogLeNet (P) | 99.63%         | 82.87%        | 100       |0.0611
| ResNet18(P)  | 99.49%         | 86.01%        | 100       |0.2817
| ResNet34(P)  | 83.63%         | 79.50%        | 1         |0.5837
| DenseNet121  | XX.XX%         | XX.XX%        | 0         |
